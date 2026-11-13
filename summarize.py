"""Summarize runs into paper-style tables (Table 1 main results, Table 2 ablation)."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedcma": "FedCMA (ours)",
         "fedcma_wo_prior": "FedCMA w.o. prior", "fedcma_wo_cl": "FedCMA w.o. CL"}
ORDER = ["fedavg", "fedprox", "fedcma", "fedcma_wo_prior", "fedcma_wo_cl"]


def main():
    for task in ["ihm", "pheno"]:
        print(f"\n=== {task.upper()} ===")
        print(f"{'method':<20}{'metric set':<12} " +
              "  ".join(f"{k:>9}" for k in (
                  ["auc", "auprc"] if task == "ihm"
                  else ["micro_ap", "macro_ap", "micro_auc", "macro_auc"])))
        for m in ORDER:
            path = os.path.join(HERE, "runs", f"{task}_{m}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                r = json.load(f)
            b = r["best"]
            keys = ["auc", "auprc"] if task == "ihm" else [
                "micro_ap", "macro_ap", "micro_auc", "macro_auc"]
            row = "  ".join(f"{b['test_personalized'][k]:>9.4f}" for k in keys)
            print(f"{NAMES[m]:<20}{'personalized':<12} {row}  (best r{b['round']})")
            row = "  ".join(f"{b['test_global'][k]:>9.4f}" for k in keys)
            print(f"{'':<20}{'global':<12} {row}")


if __name__ == "__main__":
    main()
