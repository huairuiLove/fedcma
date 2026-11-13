"""Diagnose a saved checkpoint: test metrics split by modality availability
(complete / missing-ts / missing-txt). Uses per-client priors on client test
shards, mirroring the personalized evaluation protocol.

Usage: python diagnose.py runs/ihm_fedcma.ckpt
"""
import sys

import numpy as np
import torch

from dataset import FLDataset, collate
from main import get_cfg
from metrics import compute_metrics
from model import FedCMAModel


def main(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = get_cfg([])
    for k, v in ckpt["config"].items():
        setattr(cfg, k, v)
    device = cfg.device
    model = FedCMAModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = FLDataset(cfg.task, "test", cfg.data_dir, cfg)
    use_priors = ckpt.get("priors_clients") is not None
    K = cfg.num_clients

    ys = {k: [] for k in ["all", "complete", "miss_ts", "miss_txt"]}
    ss = {k: [] for k in ys}
    with torch.no_grad():
        for k in range(K):
            idx = np.where(ds.client == k)[0]
            if len(idx) == 0:
                continue
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(ds, idx.tolist()), 64, collate_fn=collate)
            for batch in loader:
                batch = {kk: v.to(device) for kk, v in batch.items()}
                if use_priors:
                    pc = ckpt["priors_clients"][k].to(device)
                    b = batch["irg"].size(0)
                    priors = (pc[0].unsqueeze(0).expand(b, -1),
                              pc[1].unsqueeze(0).expand(b, -1))
                    p = model.predict(batch, priors, use_priors=True)
                else:
                    p = model.predict(batch, None, use_priors=False)
                y = batch["label"].cpu().numpy()
                p = p.float().cpu().numpy()
                c = batch["complete"].bool().cpu().numpy()
                its = batch["i_ts"].bool().cpu().numpy()
                itx = batch["i_txt"].bool().cpu().numpy()
                for name, m in [("all", np.ones(len(y), bool)), ("complete", c),
                                ("miss_ts", ~its), ("miss_txt", ~itx)]:
                    ys[name].append(y[m])
                    ss[name].append(p[m])
    print(f"== {ckpt_path} ({'priors' if use_priors else 'zero-feature'}) ==")
    for name in ys:
        y, p = np.concatenate(ys[name]), np.concatenate(ss[name])
        try:
            m = compute_metrics(y, p, cfg.task)
        except ValueError as e:
            m = {"error": str(e)}
        print(f"{name:10s} n={len(y):5d} {m}")


if __name__ == "__main__":
    main(sys.argv[1])
