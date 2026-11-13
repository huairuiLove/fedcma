"""Summarize ONLY one explicit experiment directory; never mix legacy results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"fedavg": "#5B6C8F", "fedprox": "#D18A39", "fedcma": "#16866B"}
NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedcma": "FedCMA"}
COLORS.update(fedcma_wo_prior="#BA724A", fedcma_wo_cl="#7C6CAD", fedcma_wo_both="#5B6C8F")
NAMES.update(fedcma_wo_prior="w/o prior", fedcma_wo_cl="w/o curriculum", fedcma_wo_both="w/o both")


def summarize(directory):
    directory = Path(directory)
    runs = [(path.parent, json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(directory.glob("*/result.json"))]
    if not runs:
        raise ValueError("No completed runs in this experiment directory")
    validation_only = all(result["validation_only"] for _, result in runs)
    if any(result["validation_only"] != validation_only for _, result in runs):
        raise ValueError("Do not mix development and final test runs")
    metric_key = "best_validation" if validation_only else "test_global"
    summary, per_seed = [], []
    for method in COLORS:
        group = [(p, r) for p, r in runs if r["method"] == method]
        if not group:
            continue
        row = {"method": method, "n_seeds": len(group), "metric_set": metric_key}
        for metric in ("auc", "auprc"):
            values = np.array([r[metric_key][metric] for _, r in group])
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.
        row["seconds_mean"] = float(np.mean([r["elapsed_seconds"] for _, r in group]))
        summary.append(row)
        for _, result in group:
            per_seed.append({"method": method, "seed": result["seed"], "best_round": result["best_round"],
                             **result[metric_key], "seconds": result["elapsed_seconds"]})
    for name, rows in (("summary.csv", summary), ("per_seed.csv", per_seed)):
        with (directory / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for method in COLORS:
        histories = [json.loads((p / "history.json").read_text(encoding="utf-8"))
                     for p, r in runs if r["method"] == method]
        if not histories:
            continue
        rounds = np.arange(1, len(histories[0]) + 1)
        for ax, group, metric, title in zip(axes, ("train_loss", "val", "val"),
                                           ("task", "auprc", "auc"),
                                           ("Training BCE", "Validation AP", "Validation AUROC")):
            values = np.array([[r[group].get("task_unweighted", r[group][metric])
                                if group == "train_loss" and metric == "task" else r[group][metric]
                                for r in h] for h in histories])
            avg, std = values.mean(0), values.std(0, ddof=1) if len(values) > 1 else np.zeros(len(rounds))
            ax.plot(rounds, avg, marker="o", label=NAMES[method], color=COLORS[method])
            ax.fill_between(rounds, avg - std, avg + std, color=COLORS[method], alpha=.12)
            ax.set(title=title, xlabel="Communication round", xticks=rounds)
            ax.grid(alpha=.18)
    axes[-1].legend(fontsize=9)
    fig.savefig(directory / "training_curves.png", dpi=180)
    fig.savefig(directory / "training_curves.pdf")
    plt.close(fig)
    if not validation_only:
        fig, axes = plt.subplots(1, 2, figsize=(max(9, len(summary) * 2.8), 4), layout="constrained")
        for ax, metric, title in zip(axes, ("auc", "auprc"), ("Test AUROC", "Test AP")):
            for i, row in enumerate(summary):
                ax.bar(i, row[f"{metric}_mean"], yerr=row[f"{metric}_std"], capsize=5,
                       color=COLORS[row["method"]], width=.6)
                ax.text(i, row[f"{metric}_mean"] + row[f"{metric}_std"] + .012,
                        f"{row[f'{metric}_mean']:.4f}", ha="center", fontsize=10)
            ax.set(title=title, xticks=range(len(summary)),
                   xticklabels=[NAMES[r["method"]] for r in summary], ylim=(0, 1))
            ax.grid(axis="y", alpha=.18)
        fig.suptitle("Mean and sample SD across training seeds")
        fig.savefig(directory / "test_comparison.png", dpi=180)
        fig.savefig(directory / "test_comparison.pdf")
        plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
