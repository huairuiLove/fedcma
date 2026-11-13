"""Federated simulation for FedCMA + baselines (FedAvg, FedProx, ablations).

Round t:
  server broadcasts global backbone theta_t (and reference priors for FedCMA)
  each client filters its samples to the easiest rho_t fraction (curriculum),
  trains E local epochs, uploads theta_k and its local priors
  server: FedAvg over theta_k; priors -> reference pbar (Eq.7); clients mix
  local priors with pbar via delta_t (Eq.8)

Baselines / ablations (cfg.method):
  fedavg   : missing modality -> zero feature, no priors/generators
  fedprox  : fedavg + proximal term mu/2 ||theta - theta_global||^2
  fedcma   : full method
  fedcma_wo_prior : curriculum only (missing -> zero feature)
  fedcma_wo_cl    : priors only (rho_t = 1 all rounds)
"""
import copy
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FLDataset, collate
from metrics import compute_metrics, weighted_average
from model import FedCMAModel


def priors_for(model, batch, p_ts, p_txt, device):
    b = batch["irg"].size(0)
    return (p_ts.unsqueeze(0).expand(b, -1).to(device),
            p_txt.unsqueeze(0).expand(b, -1).to(device))


@torch.no_grad()
def predict_all(model, loader, device, task, p_ts=None, p_txt=None):
    """Returns (y, scores) over the loader; p_ts None -> zero-feature handling."""
    model.eval()
    ys, ss = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if p_ts is None:
            priors = (batch["i_ts"] * 0, batch["i_txt"] * 0)
            scores = model.predict(batch, priors, use_priors=False)
        else:
            priors = priors_for(model, batch, p_ts, p_txt, device)
            scores = model.predict(batch, priors, use_priors=True)
        ys.append(batch["label"].cpu().numpy())
        ss.append(scores.float().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ss)


def evaluate(model, loader, device, task, p_ts=None, p_txt=None):
    y, s = predict_all(model, loader, device, task, p_ts, p_txt)
    return compute_metrics(y, s, task)


def run_fed(cfg, seed=42, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = cfg.device
    data_dir = cfg.data_dir
    use_priors = cfg.method in ("fedcma", "fedcma_wo_cl")
    use_cl = cfg.method in ("fedcma", "fedcma_wo_prior")
    use_prox = cfg.method == "fedprox"

    loaders, datasets = {}, {}
    for split in ["train", "val", "test"]:
        ds = FLDataset(cfg.task, split, data_dir, cfg)
        g = torch.Generator().manual_seed(seed if split == "train" else seed + 1)
        loaders[split] = DataLoader(ds, cfg.batch_size, shuffle=(split == "train"),
                                    collate_fn=collate, generator=g if split == "train" else None)
        datasets[split] = ds

    train_ds = datasets["train"]
    difficulty = train_ds.difficulty(cfg.xi)
    K = cfg.num_clients

    global_model = FedCMAModel(cfg).to(device)
    d = cfg.embed_dim
    priors = [torch.randn(2, d, device=device) * 0.02 for _ in range(K)]

    history = []
    best = {"val": -1.0}
    t0 = time.time()
    for t in range(cfg.rounds):
        rho_t = min(1.0, cfg.rho0 + (1 - cfg.rho0) * t / max(cfg.rounds - 1, 1)) if use_cl else 1.0
        delta_cap = getattr(cfg, "delta_max", 1.0)
        delta_t = min(delta_cap, cfg.delta0 + (delta_cap - cfg.delta0)
                      * t / max(cfg.rounds - 1, 1))

        local_states, local_priors, local_n = [], [], []
        for k in range(K):
            model = copy.deepcopy(global_model)
            model.train()
            p_ts = priors[k][0].clone().requires_grad_(True)
            p_txt = priors[k][1].clone().requires_grad_(True)
            groups = [{"params": model.parameters(), "lr": cfg.lr}]
            if use_priors:
                groups.append({"params": [p_ts, p_txt], "lr": cfg.lr})
            opt = torch.optim.Adam(groups)
            theta0 = {n: p.detach().clone() for n, p in global_model.named_parameters()}

            idx_k = np.where(train_ds.client == k)[0]
            if use_cl and rho_t < 1.0 and len(idx_k) > 1:
                thr = np.quantile(difficulty[idx_k], rho_t)
                idx_k = idx_k[difficulty[idx_k] <= thr]
            n_k = len(idx_k)

            if n_k > 0:
                for ep in range(cfg.local_epochs):
                    g = torch.Generator().manual_seed(
                        seed * 100000 + t * 1000 + k * 10 + ep)
                    perm = torch.randperm(n_k, generator=g)
                    for start in range(0, n_k, cfg.batch_size):
                        sel = idx_k[perm[start:start + cfg.batch_size].numpy()]
                        batch = collate([train_ds[int(i)] for i in sel])
                        batch = {kk: v.to(device) for kk, v in batch.items()}
                        priors_b = priors_for(model, batch, p_ts, p_txt, device) if use_priors else None
                        logits, h_true = model(batch, priors_b, use_priors=use_priors)
                        loss, parts = model.loss(logits, batch, h_true, priors_b, use_priors=use_priors)
                        if use_prox:
                            prox = sum(((p - theta0[n]).pow(2).sum())
                                       for n, p in model.named_parameters())
                            loss = loss + cfg.mu / 2 * prox
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

            local_states.append({n: p.detach().clone() for n, p in model.named_parameters()})
            local_priors.append(torch.stack([p_ts.detach(), p_txt.detach()]))
            local_n.append(max(n_k, 1))

        # server aggregation (state_dict keeps non-parameter buffers)
        w = torch.tensor(local_n, dtype=torch.float)
        w = w / w.sum()
        agg = {n: sum(w[k].item() * local_states[k][n] for k in range(K))
               for n in local_states[0]}
        sd = global_model.state_dict()
        sd.update(agg)
        global_model.load_state_dict(sd)

        if use_priors:
            pbar = sum(w[k] * local_priors[k] for k in range(K))  # Eq.7
            for k in range(K):  # Eq.8: keep local personalization
                priors[k] = (1 - delta_t) * local_priors[k] + delta_t * pbar

        # evaluation
        ref = (pbar[0], pbar[1]) if use_priors else (None, None)
        val_metrics = evaluate(global_model, loaders["val"], device, cfg.task, *ref)

        # personalized inference: each client predicts its own test shard with
        # its own priors; metrics are computed pooled over all shards (avoids
        # undefined AUC on single-label client shards under strong heterogeneity)
        ys, ss = [], []
        for k in range(K):
            idx = np.where(datasets["test"].client == k)[0]
            if len(idx) == 0:
                continue
            sub = torch.utils.data.Subset(datasets["test"], idx.tolist())
            loader = DataLoader(sub, cfg.batch_size, collate_fn=collate)
            pk = (priors[k][0], priors[k][1]) if use_priors else (None, None)
            y, s = predict_all(global_model, loader, device, cfg.task, *pk)
            ys.append(y)
            ss.append(s)
        pers_metrics = compute_metrics(np.concatenate(ys), np.concatenate(ss),
                                       cfg.task)
        glob_metrics = evaluate(global_model, loaders["test"], device, cfg.task, *ref)

        sel = "auprc" if cfg.task == "ihm" else "micro_ap"
        rec = {"round": t, "rho_t": round(rho_t, 3), "delta_t": round(delta_t, 3),
               "val": val_metrics, "test_personalized": pers_metrics,
               "test_global": glob_metrics, "min": (time.time() - t0) / 60}
        history.append(rec)
        log(f"[{cfg.method}/{cfg.task}] r{t} rho={rho_t:.2f} delta={delta_t:.2f} "
            f"val={val_metrics} test_pers[{sel}]={pers_metrics[sel]:.4f} "
            f"({rec['min']:.1f}m)")
        if val_metrics[sel] > best["val"]:
            best = {"val": val_metrics[sel], "round": t,
                    "test_personalized": pers_metrics, "test_global": glob_metrics}
            if getattr(cfg, "save_ckpt", False):
                best["ckpt"] = {
                    "config": vars(cfg),
                    "model": {k: v.cpu().clone() for k, v in global_model.state_dict().items()},
                    "priors_ref": pbar.cpu().clone() if use_priors else None,
                    "priors_clients": torch.stack([p.cpu() for p in priors]) if use_priors else None,
                }

    if getattr(cfg, "save_ckpt", False) and "ckpt" in best:
        ckpt_path = os.path.join(cfg.out_dir,
                                 f"{cfg.task}_{cfg.method}"
                                 + (f"_{cfg.tag}" if getattr(cfg, 'tag', '') else "")
                                 + ".ckpt")
        torch.save(best["ckpt"], ckpt_path)
        best["ckpt_path"] = ckpt_path
    best.pop("ckpt", None)
    return {"history": history, "best": best}
