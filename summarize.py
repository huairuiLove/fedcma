"""Summarize runs into paper-style tables.

Main table: runs/{task}_{method}_f_*.json (final batch, pooled personalized
metrics, both methods share masks/partition per setting).
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

NAMES = {"fedavg": "FedAvg", "fedcma": "FedCMA (ours)"}
SETTINGS = {"f_k5": "K=5 a=0.5", "f_k8": "K=8 a=0.5",
            "f_a01": "K=5 a=0.1", "f_k8a01": "K=8 a=0.1",
            "f_k8la2": "K=8 a=0.5 la=2.0 dm=0.4", "f_k5la2": "K=5 la=2.0 dm=0.4"}


def load(path):
    with open(path) as f:
        return json.load(f)["best"]


def main():
    for task in ["ihm", "pheno"]:
        print(f"\n=== {task.upper()} (test, personalized pooled, best-val round) ===")
        keys = ["auc", "auprc"] if task == "ihm" else [
            "micro_ap", "macro_ap", "micro_auc", "macro_auc"]
        for tag, label in SETTINGS.items():
            row = {}
            for m in ["fedavg", "fedcma"]:
                p = os.path.join(HERE, "runs", f"{task}_{m}_{tag}.json")
                if os.path.exists(p):
                    row[m] = load(p)["test_personalized"]
            if "fedavg" not in row and "fedcma" not in row:
                continue
            print(f"[{label}]")
            for m in ["fedavg", "fedcma"]:
                if m in row:
                    vals = "  ".join(f"{k}={row[m][k]:.4f}" for k in keys)
                    print(f"  {NAMES[m]:<15} {vals}  (r{load(os.path.join(HERE,'runs',f'{task}_{m}_{tag}.json'))['round']})")
            if "fedavg" in row and "fedcma" in row:
                diffs = "  ".join(f"d{k}={row['fedcma'][k]-row['fedavg'][k]:+.4f}"
                                  for k in keys)
                print(f"  {'FedCMA-FedAvg':<15} {diffs}")


if __name__ == "__main__":
    main()
