"""Report all fourth-batch trials, weight invariants and paired AP outcomes."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_prepared
from fed import save_json
from report_batch2 import read, write_csv, mean_sd, LABELS


def report(directory):
    directory = Path(directory).resolve()
    protocol, selection = read(directory / "protocol.json"), read(directory / "candidate_selection.json")
    verification, controls = read(directory / "verification.json"), read(directory / "control_verification.json")
    preservation = read(directory / "prior_batches_preservation_after.json")
    retraining, execution = read(directory / "retraining_verification.json"), read(directory / "execution_summary.json")
    assert all(r["all_passed"] for r in (verification, controls, preservation, retraining))
    third = Path(protocol["reference_experiment"])
    second = Path(read(third / "protocol.json")["reference_experiment"])
    second_protocol = read(second / "protocol.json")
    first, first_ablation = Path(second_protocol["reference_experiment"]), Path(second_protocol["reference_ablation"])
    seeds, methods = protocol["seeds"], protocol["methods"]
    results, comparisons, paired = {}, [], []
    for method in methods:
        for seed in seeds:
            roots = [first_ablation if method.startswith("fedcma_wo_") else first, second, third, directory]
            row = {"method": method, "seed": seed}
            for batch, root in enumerate(roots, 1):
                r = read(root / f"{method}_s{seed}" / "result.json")
                results[batch, method, seed] = r
                row.update({f"batch{batch}_ap": r["test_global"]["auprc"],
                            f"batch{batch}_auc": r["test_global"]["auc"], f"batch{batch}_best_round": r["best_round"]})
            row.update(batch4_minus_batch3_ap=row["batch4_ap"] - row["batch3_ap"],
                       batch4_minus_batch3_auc=row["batch4_auc"] - row["batch3_auc"])
            comparisons.append(row)
    write_csv(directory / "four_batch_comparison.csv", comparisons)
    for seed in seeds:
        a, b = results[4, "fedcma", seed], results[4, "fedcma_wo_cl", seed]
        paired.append({"seed": seed, "full_ap": a["test_global"]["auprc"], "no_cl_ap": b["test_global"]["auprc"],
                       "delta_ap": a["test_global"]["auprc"] - b["test_global"]["auprc"],
                       "full_auc": a["test_global"]["auc"], "no_cl_auc": b["test_global"]["auc"],
                       "delta_auc": a["test_global"]["auc"] - b["test_global"]["auc"],
                       "full_best_round": a["best_round"], "no_cl_best_round": b["best_round"]})
    write_csv(directory / "full_vs_no_curriculum.csv", paired)
    train = load_prepared(directory / "data" / "prepared.pt", "cpu")[0]["train"]
    labels, masks = train["label"].numpy(), train["masks"].numpy()
    missing = masks.sum(1) < 2
    weights_summary = []
    for cfg in protocol["runs"]:
        run = directory / f"{cfg['method']}_s{cfg['seed']}"
        for record in read(run / "history.json"):
            t = record["round"]
            row = {"method": cfg["method"], "seed": cfg["seed"], "round": t,
                   "phase": record["curriculum"]["phase"], "strength": record["curriculum"].get("weight_strength", 0.),
                   "draws": 0, "positive_draws": 0, "missing_positive_draws": 0,
                   "weight_mass": 0., "positive_weight_mass": 0., "missing_positive_weight_mass": 0.,
                   "weight_min": 1., "weight_max": 1., "changed_positions": 0,
                   "task_bce_unweighted": record["train_loss"]["task_unweighted"],
                   "task_bce_weighted": record["train_loss"]["task"]}
            with np.load(run / f"selection_round_{t:02d}.npz") as data:
                for k in range(cfg["num_clients"]):
                    for e in range(1, cfg["local_epochs"] + 1):
                        prefix = f"client_{k}_epoch_{e}"
                        draws, baseline, weights = data[prefix + "_train"], data[prefix + "_random"], data[prefix + "_weights"].astype(float)
                        row["draws"] += len(draws)
                        row["positive_draws"] += int(labels[draws].sum())
                        row["missing_positive_draws"] += int((labels[draws] * missing[draws]).sum())
                        row["weight_mass"] += float(weights.sum())
                        row["positive_weight_mass"] += float((weights * labels[draws]).sum())
                        row["missing_positive_weight_mass"] += float((weights * labels[draws] * missing[draws]).sum())
                        row["weight_min"] = min(row["weight_min"], float(weights.min()))
                        row["weight_max"] = max(row["weight_max"], float(weights.max()))
                        row["changed_positions"] += int((draws != baseline).sum())
            assert row["changed_positions"] == 0
            assert row["positive_draws"] == int(labels.sum()) * cfg["local_epochs"]
            assert row["missing_positive_draws"] == int((labels * missing).sum()) * cfg["local_epochs"]
            assert abs(row["weight_mass"] - row["draws"]) < .01
            assert abs(row["positive_weight_mass"] - row["positive_draws"]) < .01
            assert abs(row["missing_positive_weight_mass"] - row["missing_positive_draws"]) < .01
            weights_summary.append(row)
    write_csv(directory / "training_weight_summary.csv", weights_summary)
    gains = np.array([r["delta_ap"] for r in paired])
    wins = int((gains > 0).sum())
    conclusions = {"test_ap_positive_gain_seeds": wins, "n_seeds": len(seeds),
        "full_minus_no_curriculum_ap_mean": float(gains.mean()), "full_minus_no_curriculum_ap_min": float(gains.min()),
        "full_minus_no_curriculum_auc_mean": float(np.mean([r["delta_auc"] for r in paired])),
        "all_training_indices_identical_to_control": True, "per_group_weight_mass_verified": True}
    save_json(directory / "comparison_conclusions.json", conclusions)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, metric, title in zip(axes, ("auprc", "auc"), ("Test AP", "Test AUROC")):
        for batch, color in ((1, "#9CA7B9"), (2, "#D18A39"), (3, "#16866B"), (4, "#287CC1")):
            ax.plot(seeds, [results[batch, "fedcma", s]["test_global"][metric] for s in seeds],
                    marker="o", color=color, label=f"Full: batch {batch}")
        ax.plot(seeds, [results[4, "fedcma_wo_cl", s]["test_global"][metric] for s in seeds],
                marker="s", ls="--", color="#7C6CAD", label="No curriculum (unchanged)")
        ax.set(title=title, xlabel="Training seed", xticks=seeds)
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.savefig(directory / "four_batch_paired.png", dpi=180)
    fig.savefig(directory / "four_batch_paired.pdf")
    plt.close(fig)
    winner, cfg = selection["selected"], protocol["runs"][0]
    diagnosis = read(directory / "preparation" / "analysis" / "diagnosis_summary.json")
    full_ap = [results[4, "fedcma", s]["test_global"]["auprc"] for s in seeds]
    no_cl_ap = [results[4, "fedcma_wo_cl", s]["test_global"]["auprc"] for s in seeds]
    lines = ["# 第四批实验报告：保持随机顺序的在线渐进加权课程", "",
        f"结果目录：`{directory}`。种子42、43、44，六组正式对照全部保留。", "",
        f"完整模型AP为 {mean_sd(full_ap)}，无课程组为 {mean_sd(no_cl_ap)}。平均配对差 {gains.mean():+.4f}，最差种子差 {gains.min():+.4f}，{wins}/{len(seeds)}个种子提高。",
        ("本批三个种子的AP均超过无课程组。这里的稳定性指本次固定数据上的三个训练种子。" if wins == len(seeds) else
         "本批尚未达到三个种子的AP均超过无课程组，不能将该结果表述为稳定提升。"), "",
        "## 第三批诊断与本批调整", "",
        f"第三批已保持样本覆盖，但课程轮仍改变了 {diagnosis['active_changed_position_fraction']:.2%} 的样本位置、{diagnosis['active_changed_batch_fraction']:.2%} 的小批次归属。其难度还在每次本地训练前计算一次并复用三个epoch，可能随模型更新而过时。相关记录及验证/测试差异见 `preparation/analysis/diagnosis.md`；这些是待检验的机制解释。",
        "第四批保留与无课程组完全一致的实际随机训练顺序，在每个训练小批次中，按标签×缺失模式分组，对当前BCE取平均名次（并列同名次），映射为r∈[-1,1]，使用w=1+αₜr。单例组和全相同难度组保持单位权重。计算权重时停止梯度，再将其作为常数系数乘到任务BCE上。",
        "每组的平均权重为1，全部样本都有正权重，样本频次、组别总权重和实际批次均保持一致。任务项改为加权BCE，对齐损失、先验正则、共享、聚合、模型和学习率沿用原设置。",
        f"选中预热 {cfg['warmup_rounds']} 轮，之后将α线性升至 {cfg['curriculum_weight_max']}。每次仍为5轮×3个本地epoch，46,080次样本处理、750次优化器更新；没有增加模型前向评分遍历。逐样本BCE和权重均已保存。",
        "本实现属于课程软加权修订，与早期硬阈值筛选和第三批排序不同，技术差别已在项目说明中记录。难例加权的设计参考 [Focal Loss论文](https://arxiv.org/abs/1708.02002) 的处理思路，本批使用的有界组内名次公式是独立实现。", "",
        "## 验证候选选择", "",
        "六组预设候选各跑三个种子，仅计算验证结果。选取全部种子、最佳验证AP与最后两轮平均验证AP这两项配对增益中的最小值最高者；同分再比较平均增益。选中后冻结正式方案，候选未产生测试预测。",
        "这是课程参数的专用验证搜索，额外搜索训练量单独保留；正式六组的单次样本和更新预算一致。", "",
        "| 候选 | 最差验证增益 | 综合平均验证增益 | 最佳AP提高种子数 | 末两轮AP提高种子数 | 选用 |",
        "|---|---:|---:|---:|---:|---|"]
    for c in selection["candidates"]:
        lines.append(f"| {c['name']} | {c['stability_floor']:+.4f} | {c['combined_mean_gain']:+.4f} | {c['best_ap_positive_seeds']}/3 | {c['late_ap_positive_seeds']}/3 | {'是' if c['name']==winner['name'] else ''} |")
    lines += ["", "## 全部正式结果", "", "均值 ± 三个训练种子的样本标准差，AP采用average precision。", "",
              "| 方法 | 测试 AUROC | 测试 AP |", "|---|---:|---:|"]
    for method in methods:
        auc = [results[4, method, s]["test_global"]["auc"] for s in seeds]
        ap = [results[4, method, s]["test_global"]["auprc"] for s in seeds]
        lines.append(f"| {LABELS[method]} | {mean_sd(auc)} | {mean_sd(ap)} |")
    lines += ["", "## 四批完整模型与无课程组", "",
              "| 种子 | 第一批AP | 第二批AP | 第三批AP | 第四批AP | 无课程AP | 第四批增益 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in paired:
        s = r["seed"]
        vals = " | ".join(f"{results[b,'fedcma',s]['test_global']['auprc']:.4f}" for b in range(1, 5))
        lines.append(f"| {s} | {vals} | {r['no_cl_ap']:.4f} | {r['delta_ap']:+.4f} |")
    lines += ["", "四批六组的全部AP、AUROC和最佳轮次见 `four_batch_comparison.csv`；第四批逐种子差值见 `full_vs_no_curriculum.csv`。", "",
        "## 训练量、权重与复现核验", "",
        "- 每个epoch每个样本恰好一次，每次训练正例7,440次、缺失正例3,750次；两类对应的总权重保持相同，仅组内样本权重改变。",
        "- 独立使用CPU/SciPy从保存的逐样本BCE重算全部权重，检查名次、范围、组别权重和及加权损失；所有检查通过。实际训练索引也逐批与无课程组完全相同。",
        f"- {execution['n_validation_runs']}次验证候选、{execution['n_experiment_runs']}次正式训练、1次额外完整重训均留档；训练与核验共 {execution['total_seconds_including_audits']:.1f} 秒。",
        f"- 正式训练13,500条逐步记录，18个最佳模型重放最大预测差 {max(r['maximum_prediction_difference'] for r in verification['runs']):g}。选中候选三个训练过程及额外重训的权重、先验、每样本BCE/权重和预测逐轮精确重现。",
        "- FedAvg、FedProx、无课程、去掉两模块逐轮精确重现第三批；去掉两模块与FedAvg精确相同；所有预热轮与对应无课程对照一致。",
        f"- 前三批 {preservation['files_checked']:,} 个文件、{preservation['bytes_checked']:,} 字节重新通过哈希核验，未改写。",
        "- 保存配置、源码及哈希、环境、每步加权/未加权损失、逐样本训练顺序/BCE/权重、每轮模型、最佳模型、验证与测试预测、所有汇总和图表。训练BCE曲线展示未加权BCE。",
        "- 本批仍使用前三批已查看过的患者划分，属于连续开发比较；三个训练种子的结果不等同于新患者数据上的独立验证。", "",
        "## 重跑", "", "```powershell", ".\\.venv\\Scripts\\python.exe run_batch4.py",
        f".\\.venv\\Scripts\\python.exe verify_results.py experiments\\{directory.name}", "```", "",
        "重跑默认新建时间戳目录，已有实验目录禁止覆盖。", ""]
    (directory / "batch4_report.md").write_text("\n".join(lines), encoding="utf-8")
    print({"report": str(directory / "batch4_report.md"), **conclusions}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    report(p.parse_args().directory)
