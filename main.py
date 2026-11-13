"""FedCMA reproduction entry point.

Usage:
  python main.py --task ihm --method fedcma
  python main.py --task ihm --method fedcma --smoke   # tiny smoke run
Methods: fedavg | fedprox | fedcma | fedcma_wo_prior | fedcma_wo_cl
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from fed import run_fed

TASK_SETTINGS = {
    # ihm: 48h in-hospital mortality, 2-class CE; pheno: 24h phenotyping, 25 labels
    "ihm": {"tt_max": 48, "num_out": 2, "d_ts": 17},
    "pheno": {"tt_max": 24, "num_out": 25, "d_ts": 17},
}


def get_cfg(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["ihm", "pheno"], default="ihm")
    p.add_argument("--method", default="fedcma")
    p.add_argument("--seed", type=int, default=42)
    # federation (paper Sec. 4.1)
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--dir_alpha", type=float, default=0.5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local_epochs", type=int, default=6)
    p.add_argument("--q", type=float, default=0.5, help="per-modality missing rate")
    p.add_argument("--mask_seed", type=int, default=42)
    p.add_argument("--partition_seed", type=int, default=42)
    # backbone defaults (MultimodalMIMIC run.sh)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--embed_time", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--mu", type=float, default=0.01, help="FedProx prox weight")
    # FedCMA-specific (choices documented in README; paper gives no values)
    p.add_argument("--lambda_align", type=float, default=0.5)
    p.add_argument("--lambda_reg", type=float, default=1e-3)
    p.add_argument("--lambda_align_ts", type=float, default=1.0,
                   help="relative weight of the ts side in L_align")
    p.add_argument("--align_detach", type=int, default=1,
                   help="1: align loss trains generator only (h detached); "
                        "0: also pulls encoders toward prior features")
    p.add_argument("--xi", type=float, default=0.5, help="difficulty mix")
    p.add_argument("--rho0", type=float, default=0.6, help="curriculum start fraction")
    p.add_argument("--delta0", type=float, default=0.2, help="prior mixing start")
    p.add_argument("--delta_max", type=float, default=1.0,
                   help="prior mixing cap; delta_t: delta0 -> delta_max")
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tag", default="", help="suffix appended to output filenames")
    p.add_argument("--save_ckpt", type=int, default=0,
                   help="save model+priors at best-val round to runs/*.ckpt")
    p.add_argument("--data_dir", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "data"))
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "runs"))
    cfg = p.parse_args(argv)
    ts = TASK_SETTINGS[cfg.task]
    for k, v in ts.items():
        setattr(cfg, k, v)
    if cfg.device is None:
        cfg.device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
    if cfg.smoke:
        cfg.rounds, cfg.local_epochs, cfg.num_clients = 2, 1, 2
    cfg.tag = (cfg.tag or "").strip()
    cfg.save_ckpt = bool(cfg.save_ckpt)
    return cfg


def main():
    cfg = get_cfg()
    tag = f"{cfg.task}_{cfg.method}" + (f"_{cfg.tag}" if cfg.tag else "") \
        + ("_smoke" if cfg.smoke else "")
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, tag + ".log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"config: {vars(cfg)}")
    result = run_fed(cfg, seed=cfg.seed, log=log)
    log(f"best: {json.dumps(result['best'])}")
    with open(os.path.join(cfg.out_dir, tag + ".json"), "w") as f:
        json.dump({"config": vars(cfg), **result}, f, indent=1)


if __name__ == "__main__":
    main()
