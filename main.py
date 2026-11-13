"""Run one auditable local IHM experiment; run_experiments.py runs the suite."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path
import torch
from methods import METHODS


def get_cfg(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=list(METHODS), default="fedcma")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--prior_hidden", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda_align", type=float, default=0.001)
    p.add_argument("--lambda_reg", type=float, default=0.001)
    p.add_argument("--align_detach", action="store_true")
    p.add_argument("--xi", type=float, default=0.7)
    p.add_argument("--rho0", type=float, default=0.5)
    p.add_argument("--curriculum_schedule", choices=("legacy", "warmup_bounded", "stratified_order", "balanced_weight"), default="legacy")
    p.add_argument("--warmup_rounds", type=int, default=1)
    p.add_argument("--full_data_rounds", type=int, default=2)
    p.add_argument("--curriculum_strength", type=float, default=0.5)
    p.add_argument("--curriculum_weight_max", type=float, default=0.5)
    p.add_argument("--equal_training_budget", action="store_true")
    p.add_argument("--delta0", type=float, default=0.2)
    p.add_argument("--delta_max", type=float, default=0.6)
    p.add_argument("--grad_clip", type=float, default=5.)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--validation_only", action="store_true")
    cfg = p.parse_args(argv)
    if min(cfg.rounds, cfg.local_epochs, cfg.batch_size, cfg.num_clients, cfg.threads) <= 0:
        p.error("Training counts must be positive")
    if cfg.embed_dim % cfg.num_heads:
        p.error("embed_dim must be divisible by num_heads")
    if not all(0 <= x <= 1 for x in (cfg.xi, cfg.rho0, cfg.delta0, cfg.delta_max)):
        p.error("xi, rho0, delta0, delta_max must be in [0,1]")
    if cfg.delta0 > cfg.delta_max or cfg.lr <= 0 or min(cfg.mu, cfg.lambda_align, cfg.lambda_reg) < 0:
        p.error("Invalid optimizer or loss weights")
    if cfg.curriculum_schedule == "warmup_bounded" and not 0 <= cfg.warmup_rounds < cfg.rounds:
        p.error("warmup_rounds must be nonnegative and smaller than rounds")
    if cfg.curriculum_schedule == "stratified_order":
        if cfg.warmup_rounds < 0 or cfg.full_data_rounds < 1 or cfg.warmup_rounds + cfg.full_data_rounds >= cfg.rounds:
            p.error("Ordering curriculum needs nonnegative warmup, positive recovery, and at least one active round")
        if not 0 <= cfg.curriculum_strength <= 1:
            p.error("curriculum_strength must be in [0,1]")
    if cfg.curriculum_schedule == "balanced_weight":
        if not 0 <= cfg.warmup_rounds < cfg.rounds or not 0 <= cfg.curriculum_weight_max < 1:
            p.error("Weight curriculum requires valid warmup and curriculum_weight_max in [0,1)")
    return cfg


if __name__ == "__main__":
    from fed import run_fed
    run_fed(get_cfg())
