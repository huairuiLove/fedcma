"""Prepare one fixed cohort; default to the three methods, with optional ablations."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path
import sys

from dataset import prepare_data
from fed import run_fed, save_json
from main import get_cfg
from methods import METHODS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--train_size", type=int, default=3072)
    parser.add_argument("--subset_seed", type=int, default=2026)
    parser.add_argument("--mask_seed", type=int, default=42)
    parser.add_argument("--partition_seed", type=int, default=42)
    parser.add_argument("--missing_rate", type=float, default=0.5)
    parser.add_argument("--dir_alpha", type=float, default=0.5)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--methods", nargs="+", choices=list(METHODS),
                        default=["fedavg", "fedprox", "fedcma"])
    args, common = parser.parse_known_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    prepared = args.prepared or args.out_dir / "data" / "prepared.pt"
    configs = [get_cfg(["--prepared", str(prepared), "--out_dir", str(args.out_dir / f"{method}_s{seed}"),
                        "--num_clients", str(args.num_clients), "--method", method, "--seed", str(seed), *common])
               for seed in args.seeds for method in args.methods]
    save_json(args.out_dir / "protocol.json", {
        "command": [sys.executable, *sys.argv], "task": "IHM 48h in-hospital mortality",
        "primary_metric": "global_test_average_precision_at_best_validation_AP_round",
        "seeds": args.seeds, "methods": args.methods,
        "ablations": any(method.startswith("fedcma_wo_") for method in args.methods),
        "comparison": "same backbone, initialization per seed, data, masks, clients, optimizer and LR",
        "test_policy": "test is evaluated once after validation-based checkpoint selection",
        "configuration_fixed_before_training": True,
        "runs": [{k: str(v) if isinstance(v, Path) else v for k, v in vars(c).items()} for c in configs]})
    if not args.prepared:
        manifest = prepare_data(args.data_dir, prepared.parent, args.train_size, args.subset_seed,
                                args.mask_seed, args.partition_seed, args.missing_rate,
                                args.num_clients, args.dir_alpha)
        print({s: {k: v for k, v in meta.items() if k in ("n", "positives", "mask_counts", "clients")}
               for s, meta in manifest["splits"].items()}, flush=True)
    results = []
    for cfg in configs:
        results.append(run_fed(cfg))
        save_json(args.out_dir / "completed_runs.json", results)
    from summarize import summarize
    summarize(args.out_dir)


if __name__ == "__main__":
    main()
