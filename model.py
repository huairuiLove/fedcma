"""Lightweight patent implementation with genuine two-token modality attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from methods import method_components


class ModalityFusion(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.pool = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.modality_embedding = nn.Parameter(torch.randn(2, dim) * 0.02)

    def forward(self, features, return_weights=False):
        x = features + self.modality_embedding
        b, tokens, d = x.shape
        q, k, v = self.qkv(x).reshape(b, tokens, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // self.heads), dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(b, tokens, d)
        x = self.norm1(x + self.dropout(self.proj(attended)))
        x = self.norm2(x + self.ff(x))
        pool_weights = torch.softmax(self.pool(x), dim=1)
        result = (x * pool_weights).sum(dim=1)
        return (result, weights, pool_weights) if return_weights else result


class FedCMAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg.embed_dim, cfg.hidden_dim
        self.cfg = cfg
        self.ts_conv = nn.Sequential(nn.Conv1d(34, h, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(h, h, 3, padding=2, dilation=2), nn.ReLU())
        self.ts_proj = nn.Sequential(nn.Linear(2 * h + 102, d), nn.ReLU(), nn.LayerNorm(d))
        self.text_proj = nn.Sequential(nn.Linear(768, d), nn.ReLU(), nn.Dropout(cfg.dropout),
                                       nn.Linear(d, d), nn.LayerNorm(d))
        self.fusion = ModalityFusion(d, cfg.num_heads, cfg.dropout)
        self.classifier = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(),
                                       nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1))
        # Created last: common backbone initialization is identical across methods.
        self.generators = nn.ModuleList([
            nn.Sequential(nn.Linear(d, cfg.prior_hidden), nn.ReLU(),
                          nn.Linear(cfg.prior_hidden, d)) for _ in range(2)])
        if not method_components(cfg.method)["prior"]:
            self.generators.requires_grad_(False)

    def encode(self, batch):
        sequence = self.ts_conv(batch["ts"].transpose(1, 2))
        ts = self.ts_proj(torch.cat([sequence.mean(dim=2), sequence[:, :, -1], batch["summary"]], dim=1))
        text = self.text_proj(batch["text"])
        return torch.stack([ts, text], dim=1)

    def generated(self, priors):
        return torch.stack([g(priors[m]) for m, g in enumerate(self.generators)])

    def forward(self, batch, priors=None):
        real = self.encode(batch)
        present = batch["masks"].unsqueeze(-1)
        replacement = self.generated(priors).unsqueeze(0) if priors is not None else torch.zeros_like(real)
        features = present * real + (1 - present) * replacement
        logits = self.classifier(self.fusion(features)).squeeze(-1)
        return logits, real

    def loss(self, logits, batch, real, priors=None, sample_weights=None):
        if sample_weights is None:
            task = F.binary_cross_entropy_with_logits(logits, batch["label"])
            unweighted = task
        else:
            per_sample = F.binary_cross_entropy_with_logits(logits, batch["label"], reduction="none")
            task = (sample_weights * per_sample).mean()
            unweighted = per_sample.mean()
        align, reg = task.new_zeros(()), task.new_zeros(())
        if priors is not None:
            complete = batch["masks"].sum(dim=1) == 2
            if complete.any():
                target = real[complete].detach() if self.cfg.align_detach else real[complete]
                # Euclidean squared norm: SUM over features, mean over samples/modalities.
                align = (target - self.generated(priors).unsqueeze(0)).square().sum(-1).mean()
            reg = priors.square().sum()
        loss = task + self.cfg.lambda_align * align + self.cfg.lambda_reg * reg
        return loss, {"task": task, "task_unweighted": unweighted, "align": align, "reg": reg}
