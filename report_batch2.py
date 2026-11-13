"""Create the complete, unfiltered batch 2 comparison from saved predictions."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_prepared
from fed import save_json
from summarize import NAMES


LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedcma": "完整 FedCMA",
          "fedcma_wo_prior": "去掉先验", "fedcma_wo_cl": "去掉课程学习", "fedcma_wo_both": "两者都去掉"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"


def report(directory):
    directory = Path(directory).resolve()
    protocol = read(directory / "protocol.json")
    verification = read(directory / "verification.json")
    controls = read(directory / "control_verification.json")
    retraining = read(directory / "retraining_verification.json")
    preservation = read(directory / "batch1_preservation_after.json")
    assert all(r["all_passed"] for r in (verification, controls, retraining, preservation))
    old_main, old_ablation = Path(protocol["reference_experiment"]), Path(protocol["reference_ablation"])
    seeds, methods = protocol["seeds"], protocol["methods"]
    new_results, old_results, comparisons = {}, {}, []
    for method in methods:
        old_dir = old_ablation if method.startswith("fedcma_wo_") else old_main
        for seed in seeds:
            new_run, old_run = directory / f"{method}_s{seed}", old_dir / f"{method}_s{seed}"
            new, old = read(new_run / "result.json"), read(old_run / "result.json")
            new_results[method, seed], old_results[method, seed] = new, old
            old_history = read(old_run / "history.json")
            comparisons.append({"method": method, "seed": seed,
                "batch1_best_round": old["best_round"], "batch2_best_round": new["best_round"],
                "batch1_auc": old["test_global"]["auc"], "batch2_auc": new["test_global"]["auc"],
                "delta_auc": new["test_global"]["auc"] - old["test_global"]["auc"],
                "batch1_ap": old["test_global"]["auprc"], "batch2_ap": new["test_global"]["auprc"],
                "delta_ap": new["test_global"]["auprc"] - old["test_global"]["auprc"],
                "batch1_examples": sum(r["examples_processed"] for r in old_history),
                "batch2_examples": new["training_budget"]["examples_processed"],
                "batch1_steps": sum(r["optimizer_steps"] for r in old_history),
                "batch2_steps": new["training_budget"]["optimizer_steps"]})
    write_csv(directory / "batch1_vs_batch2.csv", comparisons)
    paired = []
    for seed in seeds:
        full, no_cl = new_results["fedcma", seed], new_results["fedcma_wo_cl", seed]
        paired.append({"seed": seed, "full_best_round": full["best_round"], "no_cl_best_round": no_cl["best_round"],
                       "full_auc": full["test_global"]["auc"], "no_cl_auc": no_cl["test_global"]["auc"],
                       "delta_auc": full["test_global"]["auc"] - no_cl["test_global"]["auc"],
                       "full_ap": full["test_global"]["auprc"], "no_cl_ap": no_cl["test_global"]["auprc"],
                       "delta_ap": full["test_global"]["auprc"] - no_cl["test_global"]["auprc"]})
    write_csv(directory / "full_vs_no_curriculum.csv", paired)
    train = load_prepared(directory / "data" / "prepared.pt", "cpu")[0]["train"]
    labels, masks = train["label"].numpy(), train["masks"].numpy()
    diagnostics = []
    for settings in protocol["runs"]:
        run = directory / f"{settings['method']}_s{settings['seed']}"
        history = read(run / "history.json")
        for rec in history:
            with np.load(run / f"selection_round_{rec['round']:02d}.npz") as data:
                for k in range(settings["num_clients"]):
                    full, pool = data[f"client_{k}_all"], data[f"client_{k}_selected"]
                    draws = np.concatenate([data[f"client_{k}_epoch_{e}_train"]
                                            for e in range(1, settings["local_epochs"] + 1)])
                    diagnostics.append({"method": settings["method"], "seed": settings["seed"],
                        "round": rec["round"], "client": k, "phase": rec["curriculum"]["phase"], "rho": rec["rho"],
                        "full_n": len(full), "pool_n": len(pool), "training_draws": len(draws),
                        "full_positives": int(labels[full].sum()), "pool_positives": int(labels[pool].sum()),
                        "draw_positives": int(labels[draws].sum()),
                        "full_incomplete": int((masks[full].sum(1) < 2).sum()),
                        "pool_incomplete": int((masks[pool].sum(1) < 2).sum()),
                        "draw_incomplete": int((masks[draws].sum(1) < 2).sum())})
    write_csv(directory / "sampling_diagnostics.csv", diagnostics)
    round_stats = []
    for t in range(1, protocol["runs"][0]["rounds"] + 1):
        groups = [[r for r in diagnostics if r["method"] == "fedcma" and r["seed"] == s and r["round"] == t] for s in seeds]
        round_stats.append({"round": t, "rho": groups[0][0]["rho"], "phase": groups[0][0]["phase"],
                            "eligible": [sum(r["pool_n"] for r in g) for g in groups],
                            "draw_positive_rate": [sum(r["draw_positives"] for r in g) / sum(r["training_draws"] for r in g)
                                                   for g in groups]})
    save_json(directory / "curriculum_diagnostics.json", {"seeds": seeds, "full_model_rounds": round_stats,
              "original_positive_rate": float(labels.mean())})

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.3), layout="constrained")
    x = np.arange(len(methods))
    for ax, metric, title in zip(axes, ("auc", "auprc"), ("Test AUROC", "Test AP")):
        for offset, results, color, label in ((-.18, old_results, "#9CA7B9", "Batch 1"),
                                               (.18, new_results, "#16866B", "Batch 2")):
            values = np.array([[results[m, s]["test_global"][metric] for s in seeds] for m in methods])
            ax.bar(x + offset, values.mean(1), .34, yerr=values.std(1, ddof=1), capsize=3, color=color, label=label)
        ax.set(xticks=x, xticklabels=[NAMES[m] for m in methods], title=title, ylim=(0, 1))
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=.15)
        ax.legend()
    fig.suptitle("Fixed data, three paired training seeds; mean and sample SD")
    fig.savefig(directory / "batch_comparison.png", dpi=180)
    fig.savefig(directory / "batch_comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for i, seed in enumerate(seeds):
        axes[0].plot([r["round"] for r in round_stats], [r["eligible"][i] / len(labels) for r in round_stats], marker="o", label=f"seed {seed}")
        axes[1].plot([r["round"] for r in round_stats], [r["draw_positive_rate"][i] for r in round_stats], marker="o", label=f"seed {seed}")
    axes[0].set(title="FedCMA: eligible share", ylabel="Eligible / all training samples", ylim=(0, 1.05))
    axes[1].set(title="FedCMA: positive share in training draws", ylabel="Positive draws / all draws")
    axes[1].axhline(float(labels.mean()), color="black", ls="--", lw=1, label="Original training rate")
    for ax in axes:
        ax.set(xlabel="Communication round", xticks=[r["round"] for r in round_stats])
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.savefig(directory / "curriculum_diagnostics.png", dpi=180)
    fig.savefig(directory / "curriculum_diagnostics.pdf")
    plt.close(fig)

    full_comparisons = [r for r in comparisons if r["method"] == "fedcma"]
    improvement = {metric: float(np.mean([r[f"delta_{metric}"] for r in full_comparisons])) for metric in ("auc", "ap")}
    paired_mean = {metric: float(np.mean([r[f"delta_{metric}"] for r in paired])) for metric in ("auc", "ap")}
    budget = new_results["fedcma", seeds[0]]["training_budget"]
    execution = read(directory / "execution_summary.json")
    lines = ["# 第二批实验报告：预热、阈值调度与相同训练量", "",
             f"实验目录：`{directory}`。第一批原路径和全部内容保留；批次索引见 `experiments/batch_01/manifest.json`。", "",
             f"第二批完整模型相对第一批：AUROC 平均变化 {improvement['auc']:+.4f}，AP 平均变化 {improvement['ap']:+.4f}。",
             f"第二批完整模型相对去掉课程学习：AUROC 平均差 {paired_mean['auc']:+.4f}，AP 平均差 {paired_mean['ap']:+.4f}。",
             f"完整模型 AP 高于无课程组的种子数：{sum(r['delta_ap'] > 0 for r in paired)}/{len(seeds)}。这三次训练使用同一份数据划分。", "",
             "## 固定实验方案", "",
             "- 沿用第一批的训练 3,072 / 验证 427 / 测试 1,131 个样本、客户端分配、缺失掩码、轻量模型、损失权重和优化器。三个训练种子为 42、43、44。",
             "- 总计仍为 5 轮，每轮本地 3 个 epoch；第 1 轮全样本预热，包含在这 5 轮内。",
             "- 难度仍为 S = 0.7P + 0.3C，P 为缺失模态比例，C 为当前模型 BCE 除以当前评分批次最大 BCE。",
             "- 仅由训练掩码计算上界：P 最大 0.5，S 最大 0.65。第 2—5 轮阈值为 0.55、0.583333、0.616667、0.65；比较时加 1e-7 浮点容差。",
             "- 入选池每个 epoch 随机排列，用完后再排列，重复到原始客户端 n_k 次样本处理；池内各样本重复次数最多相差 1。无课程组仍为全体样本的一次随机排列。",
             f"- 六组每次训练均处理 {budget['examples_processed']:,} 次样本，执行 {budget['optimizer_steps']:,} 次优化器更新，逐客户端、逐 epoch 的批次数和批大小也一致。课程评分另有前向计算开销，已单独记录。",
             "- 第二批未新增参数搜索。方案在训练前写入 protocol.json，最佳轮次仅由验证 AP 选择；全部测试结果均保留。", "",
             "## 第二批全部结果", "", "均值 ± 三个训练种子的样本标准差。AP 使用 average precision。", "",
             "| 方法 | 测试 AUROC | 测试 AP |", "|---|---:|---:|"]
    for method in methods:
        auc = [new_results[method, s]["test_global"]["auc"] for s in seeds]
        ap = [new_results[method, s]["test_global"]["auprc"] for s in seeds]
        lines.append(f"| {LABELS[method]} | {mean_sd(auc)} | {mean_sd(ap)} |")
    lines += ["", "## 完整模型逐种子对照", "",
              "| 种子 | 第一批 AP | 第二批 AP | 无课程 AP | 第二批完整 − 无课程 AP | 完整 / 无课程最佳轮次 |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        old_ap = old_results["fedcma", r["seed"]]["test_global"]["auprc"]
        lines.append(f"| {r['seed']} | {old_ap:.4f} | {r['full_ap']:.4f} | {r['no_cl_ap']:.4f} | {r['delta_ap']:+.4f} | {r['full_best_round']} / {r['no_cl_best_round']} |")
    lines += ["", "`batch1_vs_batch2.csv` 保留六组、三个种子的全部前后差异；`full_vs_no_curriculum.csv` 保留配对 AP 和 AUROC 差异。", "",
              "## 课程筛选与实际训练分布", "",
              "| 轮次 | 阶段 | 阈值 | 入选样本数（42 / 43 / 44） | 实际训练正例比例（42 / 43 / 44） |", "|---|---|---:|---|---|"]
    for r in round_stats:
        counts = " / ".join(str(n) for n in r["eligible"])
        rates = " / ".join(f"{n:.2%}" for n in r["draw_positive_rate"])
        lines.append(f"| {r['round']} | {r['phase']} | {r['rho']:.6f} | {counts} | {rates} |")
    lines += ["", f"原始训练正例比例为 {float(labels.mean()):.2%}。相同训练次数保证更新预算一致，但筛选仍可能改变训练样本分布；`sampling_diagnostics.csv` 保留所有客户端的正负例、模态缺失和重复训练计数。",
              "当前实验同时改了预热、阈值和训练量，前后变化不能单独归因于其中某一个因素。", "",
              "## 可复现核验与留档", "",
              f"- 正式训练 {verification['n_runs']} 次，另完整重训一次 FedCMA seed 42 核验；总执行时间含汇总和核验为 {execution['total_seconds_including_audits']:.1f} 秒。",
              f"- 正式训练逐步记录 {sum(r['step_records'] for r in verification['runs']):,} 条；全部最佳模型重放通过，最大预测差 {max(r['maximum_prediction_difference'] for r in verification['runs']):g}。",
              "- 完整模型与无课程组的第 1 轮权重、先验及验证指标完全一致；去先验组的预热也与 FedAvg 一致。",
              "- FedAvg、FedProx、无课程和去掉两模块四组的所有轮次权重与验证/测试预测均精确重现第一批；第二批去掉两模块也精确重现第二批 FedAvg。",
              "- 额外完整重训的所有轮次权重、先验、验证/测试预测、筛选与实际训练索引均一致。",
              f"- 第一批 {preservation['files_checked']:,} 个文件逐个校验 SHA-256，通过，原文件未改动。",
              "- 每次训练保存配置、源码及哈希、依赖、环境、逐步损失、每轮指标、评分/入选/实际重复索引、每轮检查点、最佳模型和逐样本预测；报告和曲线另行保存。", "",
              "## 重跑命令", "", "在项目目录的 PowerShell 中执行：", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe run_batch2.py",
              f".\\.venv\\Scripts\\python.exe verify_results.py experiments\\{directory.name}",
              "```", "", "重跑默认新建带时间戳的第二批目录，禁止覆盖已存在输出目录。第一批仍可使用原入口或已保存源码快照重现。", ""]
    (directory / "batch2_report.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(directory / "comparison_conclusions.json", {
        "full_batch2_minus_batch1": improvement, "full_minus_no_curriculum": paired_mean,
        "ap_wins_over_no_curriculum": sum(r["delta_ap"] > 0 for r in paired), "n_seeds": len(seeds)})
    print(json.dumps({"report": str(directory / 'batch2_report.md'), "full_change": improvement,
                      "full_minus_no_curriculum": paired_mean}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    report(parser.parse_args().directory)
