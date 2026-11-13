"""Small, recorded FedCMA-only validation search; test predictions are never used.

Baseline parameters remain the common fixed settings, not separately tuned.
This asymmetric tuning budget must be disclosed in the experiment report.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import numpy as np
from main import get_cfg
from fed import run_fed, save_json


CANDIDATES = [
    {"lambda_align": .001, "rho0": .5},
    {"lambda_align": .0001, "rho0": .4},
    {"lambda_align": .0001, "rho0": .6},
    {"lambda_align": .001, "rho0": .4},
    {"lambda_align": .001, "rho0": .6},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    seeds = [42, 43, 44]
    save_json(args.out_dir / "search_plan.json", {
        "selection": "highest mean best-round validation AP across all three seeds",
        "candidates": CANDIDATES, "seeds": seeds,
        "baseline_settings": "fixed common LR=0.001, rounds=5, local_epochs=3; no baseline search",
        "test_access": "none"})
    table = []
    for i, params in enumerate(CANDIDATES):
        results = []
        for seed in seeds:
            options = ["--prepared", str(args.prepared), "--out_dir", str(args.out_dir / f"c{i}_s{seed}"),
                       "--method", "fedcma", "--seed", str(seed), "--validation_only"]
            for key, value in params.items():
                options += ["--" + key, str(value)]
            results.append(run_fed(get_cfg(options)))
        record = {"candidate": i, **params,
                  "mean_validation_ap": float(np.mean([r["best_validation"]["auprc"] for r in results])),
                  "mean_validation_auc": float(np.mean([r["best_validation"]["auc"] for r in results]))}
        table.append(record)
        save_json(args.out_dir / "search_results.json", table)
    winner = max(table, key=lambda row: row["mean_validation_ap"])
    save_json(args.out_dir / "selected.json", winner)
    print("SELECTED " + json.dumps(winner), flush=True)


if __name__ == "__main__":
    main()
