"""FedCMA model: MultimodalMIMIC-style encoders + modal-aware priors/generators
+ attention-based cross-modal fusion.

Per client k and modality m in {ts, txt}:
  h^(m)   = f_theta^(m)(x^(m))            (modality feature, 128-d)
  h~^(m)  = G^(m)(p_k^(m))  if I^(m)=0    (pseudo feature from the prior)
  z       = phi(h~^(ts), h~^(txt))        (bidirectional cross-attention fusion)
  y_hat   = g(z)

Loss (Eq. 5): L = L_task + lambda_align * L_align + lambda_reg * L_reg with
L_align = mean over complete samples of ||h^(m) - G^(m)(p_k^(m))||^2 (h detached)
L_reg   = ||p_ts||^2 + ||p_txt||^2.
"""
import torch
import torch.nn.functional as F
from torch import nn

from module import (TimeEmbed, TransformerEncoder, gateMLP, multiTimeAttention)


class ModalityEncoder(nn.Module):
    """Irregular-time attention encoder for one modality stream.

    ts:  values+mask (2*17 ch) at irregular hours -> grid [B, T, d]
    txt: frozen 768-d BioClinicalBERT CLS note embeddings at irregular
         offsets -> grid [B, T, d]
    """

    def __init__(self, in_dim, d, embed_time, num_heads, layers, dropout):
        super().__init__()
        self.time_embed = TimeEmbed(embed_time)
        self.time_attn = multiTimeAttention(in_dim, d, embed_time, 8)
        self.mem = TransformerEncoder(d, num_heads, layers, dropout)
        self.d = d

    def forward(self, values, times, key_mask, query_times):
        te_key = self.time_embed(times)
        te_query = self.time_embed(query_times)
        grid = self.time_attn(te_query, te_key, values, key_mask)  # [B, T, d]
        grid = self.mem(grid)
        return grid.mean(dim=1)                                    # [B, d]


class FedCMAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        self.ts_time_embed = TimeEmbed(cfg.embed_time)
        self.ts_attn = multiTimeAttention(cfg.d_ts * 2, d, cfg.embed_time, 8)
        self.ts_mem = TransformerEncoder(d, cfg.num_heads, cfg.layers, cfg.dropout)
        self.proj_ts_reg = nn.Conv1d(cfg.d_ts * 2, d, kernel_size=1, bias=False)
        self.moe = gateMLP(2 * d, d, 1, cfg.dropout)
        self.txt_enc = ModalityEncoder(768, d, cfg.embed_time, cfg.num_heads,
                                       cfg.layers, cfg.dropout)
        self.register_buffer("time_query", torch.linspace(0, 1, cfg.tt_max))

        # modal-aware prior generators (global/shared parameters)
        self.gen_ts = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.gen_txt = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        # attention-based cross-modal fusion
        self.cross_txt_to_ts = nn.MultiheadAttention(d, cfg.num_heads,
                                                     dropout=cfg.dropout, batch_first=True)
        self.cross_ts_to_txt = nn.MultiheadAttention(d, cfg.num_heads,
                                                     dropout=cfg.dropout, batch_first=True)
        fd = 2 * d
        self.proj1 = nn.Linear(fd, fd)
        self.proj2 = nn.Linear(fd, fd)
        self.out_layer = nn.Linear(fd, cfg.num_out)

    # --- modality encoders -------------------------------------------------
    def encode_ts(self, irg, irg_mask, tt, reg_ts):
        te_key = self.ts_time_embed(tt)
        te_query = self.ts_time_embed(self.time_query.unsqueeze(0).expand(irg.size(0), -1))
        x = torch.cat([irg, irg_mask], dim=-1)
        m = torch.cat([irg_mask, irg_mask], dim=-1)
        grid_irg = self.ts_attn(te_query, te_key, x, m)                     # [B,T,d]
        grid_reg = self.proj_ts_reg(reg_ts.transpose(1, 2)).transpose(1, 2)  # [B,T,d]
        gate = self.moe(torch.cat([grid_irg, grid_reg], dim=-1))
        grid = gate * grid_irg + (1 - gate) * grid_reg
        return self.ts_mem(grid).mean(dim=1)

    def encode_txt(self, batch, note_mask):
        emb = batch["note_emb"]
        return self.txt_enc(emb, batch["note_time"], note_mask,
                            self.time_query.unsqueeze(0).expand(emb.size(0), -1))

    # --- forward ------------------------------------------------------------
    def fuse(self, h_ts, h_txt):
        a = h_ts.unsqueeze(1)
        b = h_txt.unsqueeze(1)
        h_ts_f, _ = self.cross_ts_to_txt(a, b, b)
        h_txt_f, _ = self.cross_txt_to_ts(b, a, a)
        z = torch.cat([h_ts_f.squeeze(1), h_txt_f.squeeze(1)], dim=-1)
        z = self.proj2(F.dropout(F.relu(self.proj1(z)), p=self.cfg.dropout,
                                 training=self.training)) + z
        return z

    def forward(self, batch, priors, use_priors=True):
        """priors: (p_ts [B,d], p_txt [B,d]) broadcast client priors per sample."""
        h_ts = self.encode_ts(batch["irg"], batch["irg_mask"], batch["tt"], batch["reg"])
        h_txt = self.encode_txt(batch, batch["note_mask"])
        i_ts, i_txt = batch["i_ts"].unsqueeze(-1), batch["i_txt"].unsqueeze(-1)

        if use_priors:
            p_ts, p_txt = priors
            h_ts_in = i_ts * h_ts + (1 - i_ts) * self.gen_ts(p_ts)
            h_txt_in = i_txt * h_txt + (1 - i_txt) * self.gen_txt(p_txt)
        else:
            # baseline handling of missing modalities: zero feature
            h_ts_in = i_ts * h_ts
            h_txt_in = i_txt * h_txt
        z = self.fuse(h_ts_in, h_txt_in)
        return self.out_layer(z), (h_ts, h_txt)

    def loss(self, logits, batch, h_true, priors, use_priors=True):
        cfg = self.cfg
        y = batch["label"]
        if cfg.task == "ihm":
            l_task = F.cross_entropy(logits, y)
        else:
            l_task = F.binary_cross_entropy_with_logits(logits, y.float())
        out = {"task": l_task.detach()}
        loss = l_task
        if use_priors:
            p_ts, p_txt = priors
            complete = batch["complete"].float()
            gen_ts, gen_txt = self.gen_ts(p_ts), self.gen_txt(p_txt)
            if complete.sum() > 0:
                h_ts, h_txt = h_true
                detach = bool(getattr(cfg, "align_detach", 1))
                h_ts_a = h_ts.detach() if detach else h_ts
                h_txt_a = h_txt.detach() if detach else h_txt
                # ts-side alignment can be weighted higher: ts features are
                # higher-variance than frozen-BERT text features, so the
                # generator needs a stronger pull to track them
                w_ts = float(getattr(cfg, "lambda_align_ts", 1.0))
                l_align = (complete * (w_ts * F.mse_loss(gen_ts, h_ts_a, reduction="none").mean(-1)
                                       + F.mse_loss(gen_txt, h_txt_a, reduction="none").mean(-1))).sum() \
                    / ((1 + w_ts) * complete.sum())
                loss = loss + cfg.lambda_align * l_align
                out["align"] = l_align.detach()
            l_reg = p_ts.pow(2).mean() + p_txt.pow(2).mean()
            loss = loss + cfg.lambda_reg * l_reg
            out["reg"] = l_reg.detach()
        out["total"] = loss.detach()
        return loss, out

    @torch.no_grad()
    def predict(self, batch, priors, use_priors=True):
        logits, _ = self.forward(batch, priors, use_priors=use_priors)
        if self.cfg.task == "ihm":
            return torch.softmax(logits, dim=-1)[:, 1]
        return torch.sigmoid(logits)
