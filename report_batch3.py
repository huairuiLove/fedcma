"""Report all batch 3 trials, matched controls and all three batches."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_prepared
from fed import save_json
from report_batch2 import read, write_csv, mean_sd, LABELS
from summarize import NAMES


def report(directory):
    directory = Path(directory).resolve()
    protocol, selection = read(directory / "protocol.json"), read(directory / "candidate_selection.json")
    verification, controls = read(directory / "verification.json"), read(directory / "control_verification.json")
    preservation = read(directory / "prior_batches_preservation_after.json")
    retraining, execution = read(directory / "retraining_verification.json"), read(directory / "execution_summary.json")
    assert all(r["all_passed"] for r in (verification, controls, preservation, retraining))
    second = Path(protocol["reference_experiment"])
    old_protocol = read(second / "protocol.json")
    first, first_ablation = Path(old_protocol["reference_experiment"]), Path(old_protocol["reference_ablation"])
    seeds, methods = protocol["seeds"], protocol["methods"]
    results, comparisons, paired = {}, [], []
    for method in methods:
        for seed in seeds:
            roots = [first_ablation if method.startswith("fedcma_wo_") else first, second, directory]
            row = {"method": method, "seed": seed}
            for batch, root in enumerate(roots, 1):
                r = read(root / f"{method}_s{seed}" / "result.json")
                results[batch, method, seed] = r
                row.update({f"batch{batch}_ap": r["test_global"]["auprc"],
                            f"batch{batch}_auc": r["test_global"]["auc"], f"batch{batch}_best_round": r["best_round"]})
            row["batch3_minus_batch2_ap"] = row["batch3_ap"] - row["batch2_ap"]
            row["batch3_minus_batch2_auc"] = row["batch3_auc"] - row["batch2_auc"]
            comparisons.append(row)
    write_csv(directory / "three_batch_comparison.csv", comparisons)
    for seed in seeds:
        a, b = results[3, "fedcma", seed], results[3, "fedcma_wo_cl", seed]
        paired.append({"seed": seed, "full_ap": a["test_global"]["auprc"], "no_cl_ap": b["test_global"]["auprc"],
                       "delta_ap": a["test_global"]["auprc"] - b["test_global"]["auprc"],
                       "full_auc": a["test_global"]["auc"], "no_cl_auc": b["test_global"]["auc"],
                       "delta_auc": a["test_global"]["auc"] - b["test_global"]["auc"],
                       "full_best_round": a["best_round"], "no_cl_best_round": b["best_round"]})
    write_csv(directory / "full_vs_no_curriculum.csv", paired)
    train = load_prepared(directory / "data" / "prepared.pt", "cpu")[0]["train"]
    labels, masks = train["label"].numpy(), train["masks"].numpy()
    missing = masks.sum(1) < 2
    distributions = []
    for cfg in protocol["runs"]:
        run = directory / f"{cfg['method']}_s{cfg['seed']}"
        for record in read(run / "history.json"):
            t = record["round"]
            row = {"method": cfg["method"], "seed": cfg["seed"], "round": t,
                   "phase": record["curriculum"]["phase"], "strength": record["curriculum"].get("order_strength", 0.),
                   "draws": 0, "positive_draws": 0, "missing_positive_draws": 0, "changed_positions": 0}
            with np.load(run / f"selection_round_{t:02d}.npz") as selection_file:
                for k in range(cfg["num_clients"]):
                    for e in range(1, cfg["local_epochs"] + 1):
                        draws = selection_file[f"client_{k}_epoch_{e}_train"]
                        baseline = selection_file[f"client_{k}_epoch_{e}_random"]
                        row["draws"] += len(draws)
                        row["positive_draws"] += int(labels[draws].sum())
                        row["missing_positive_draws"] += int((labels[draws] * missing[draws]).sum())
                        row["changed_positions"] += int((draws != baseline).sum())
            assert row["positive_draws"] == int(labels.sum()) * cfg["local_epochs"]
            assert row["missing_positive_draws"] == int((labels * missing).sum()) * cfg["local_epochs"]
            distributions.append(row)
    write_csv(directory / "training_distribution.csv", distributions)
    wins = sum(r["delta_ap"] > 0 for r in paired)
    gains = np.array([r["delta_ap"] for r in paired])
    auc_gains = np.array([r["delta_auc"] for r in paired])
    conclusions = {"test_ap_positive_gain_seeds": wins, "n_seeds": len(seeds),
                   "full_minus_no_curriculum_ap_mean": float(gains.mean()),
                   "full_minus_no_curriculum_ap_min": float(gains.min()),
                   "full_minus_no_curriculum_auc_mean": float(auc_gains.mean()),
                   "all_training_samples_once_per_epoch_verified": True,
                   "all_minibatch_label_mask_counts_match_control": True}
    save_json(directory / "comparison_conclusions.json", conclusions)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, metric, title in zip(axes, ("auprc", "auc"), ("Test AP", "Test AUROC")):
        for batch, color in ((1, "#9CA7B9"), (2, "#D18A39"), (3, "#16866B")):
            vals = [results[batch, "fedcma", s]["test_global"][metric] for s in seeds]
            ax.plot(seeds, vals, marker="o", color=color, label=f"Full model: batch {batch}")
        ax.plot(seeds, [results[3, "fedcma_wo_cl", s]["test_global"][metric] for s in seeds],
                marker="s", ls="--", color="#7C6CAD", label="No curriculum (identical all batches)")
        ax.set(title=title, xlabel="Training seed", xticks=seeds)
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.savefig(directory / "three_batch_paired.png", dpi=180)
    fig.savefig(directory / "three_batch_paired.pdf")
    plt.close(fig)
    winner = selection["selected"]
    settings = protocol["runs"][0]
    budget = results[3, "fedcma", seeds[0]]["training_budget"]
    prior_exposure = read(directory / "preparation" / "analysis" / "batch2_positive_exposure.json")
    lines = ["# 第三批报告：保持样本覆盖与组别分布的课程学习", "",
             f"结果目录：`{directory}`。训练种子为42、43、44，数据、基础模型、损失权重、学习率和总训练轮数保持一致。", "",
             f"第三批完整模型相对无课程组的平均 AP 差为 {gains.mean():+.4f}，三个种子中 {wins} 个提高，最差种子差为 {gains.min():+.4f}；平均 AUROC 差为 {auc_gains.mean():+.4f}。",
             ("本次三个训练种子的AP均提高；这描述当前固定数据上的结果。" if wins == len(seeds) else
              "本次尚未得到三个训练种子的AP都提高的结果，所有不利结果也保留。"), "",
             "## 第二批问题分析", "",
             "第二批总训练量相等，但困难正例尤其模态缺失正例的实际训练次数减少。每个种子的正例应训练7,440次、缺失正例应训练3,750次，实际为：", "",
             "| 种子 | 第二批正例次数 | 第二批缺失正例次数 | 第三批正例次数 | 第三批缺失正例次数 |", "|---|---:|---:|---:|---:|"]
    for r in prior_exposure:
        lines.append(f"| {r['seed']} | {r['positive_draws_actual']} | {r['missing_positive_draws_actual']} | {r['positive_draws_expected']} | {r['missing_positive_draws_expected']} |")
    lines += ["", "第二批seed 44的测试前50名正例为21个，无课程组为27个；前100名为37个对44个。AP受正例排名影响，因此整体AUROC上升时，高分段精确率/召回仍可能下降。训练曝光缺口是事实，其与AP变化的因果关系不能单靠这些记录确定。完整诊断见 `preparation/analysis/diagnosis.md`。",
              "AP按召回增量对精确率加权，定义见 [scikit-learn文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)。", "",
              "## 本批课程调整", "",
              "- 每个客户端每个epoch保留每个样本恰好一次，取消困难样本硬剔除及入选样本重复补齐。",
              "- 保留普通随机排列的每个位置所属标签×缺失模式，仅在相同组别内部按难度调整顺序。因此每个小批次的标签、模态缺失组成也与无课程组相同。",
              "- 难度沿用缺失比例与当前BCE的组合，将BCE分母改为同一客户端最大BCE；排序使用同组内的难度名次与随机名次加权，避免评分小批次的不同分母干扰排序。",
              f"- 第1轮全样本预热；选中初始排序强度 {settings['curriculum_strength']}，最后 {settings['full_data_rounds']} 轮恢复普通随机顺序。具体每轮计划写入protocol.json。",
              f"- 总计5轮×3个本地epoch；每次仍为 {budget['examples_processed']:,} 次样本处理和 {budget['optimizer_steps']:,} 次更新。每客户端、每epoch、每个样本频次及小批次组别计数均核验。评分额外前向计算未计入训练更新预算。",
              "这是第三批修订的课程实现，课程操作变为分层排序。专利中的硬阈值筛选步骤若采用本批结果，应同步说明这一实现调整；原文档未被改写。", "",
              "## 仅验证集选择设置", "",
              "预先固定四组候选，每组同样运行三个种子。以相对无课程组的最差种子验证AP增益为第一准则，平均增益为第二准则。所有候选均为validation_only，未生成测试预测；选定后才开始正式测试评价。",
              "这是课程参数专用的小范围验证搜索，基线参数沿用前两批；正式单次训练预算一致，额外搜索预算单独记录。", "",
              "| 候选 | 三种子验证 AP | 最差配对增益 | 平均配对增益 | 选用 |", "|---|---|---:|---:|---|"]
    for c in selection["candidates"]:
        vals = " / ".join(f"{v:.4f}" for v in c["validation_ap"])
        lines.append(f"| {c['name']} | {vals} | {c['minimum_gain']:+.4f} | {c['mean_gain']:+.4f} | {'是' if c['name'] == winner['name'] else ''} |")
    lines += ["", "## 第三批全部正式结果", "", "均值 ± 三个训练种子的样本标准差；AP为average precision。", "",
              "| 方法 | 测试 AUROC | 测试 AP |", "|---|---:|---:|"]
    for method in methods:
        auc = [results[3, method, s]["test_global"]["auc"] for s in seeds]
        ap = [results[3, method, s]["test_global"]["auprc"] for s in seeds]
        lines.append(f"| {LABELS[method]} | {mean_sd(auc)} | {mean_sd(ap)} |")
    lines += ["", "## 完整模型三个批次与无课程组", "",
              "| 种子 | 第一批 AP | 第二批 AP | 第三批 AP | 无课程 AP | 第三批 − 无课程 |", "|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        s = r["seed"]
        lines.append(f"| {s} | {results[1,'fedcma',s]['test_global']['auprc']:.4f} | {results[2,'fedcma',s]['test_global']['auprc']:.4f} | {r['full_ap']:.4f} | {r['no_cl_ap']:.4f} | {r['delta_ap']:+.4f} |")
    lines += ["", "六组全部种子的三批对照见 `three_batch_comparison.csv`，本批完整模型/无课程逐种子AP和AUROC差值见 `full_vs_no_curriculum.csv`。", "",
              "## 核验与留档", "",
              f"- {execution['n_validation_runs']}次验证候选、{execution['n_experiment_runs']}次正式训练、1次额外完整重训，全部保留。包括训练和核验共 {execution['total_seconds_including_audits']:.1f} 秒。",
              f"- 正式训练共 {sum(r['step_records'] for r in verification['runs']):,} 条逐步记录，18个最佳模型重放，最大预测差为 {max(r['maximum_prediction_difference'] for r in verification['runs']):g}。",
              "- 未改动的FedAvg、FedProx、无课程、去掉两模块四组逐轮精确重现第二批；去掉两模块与FedAvg精确一致。",
              "- 预热轮、与无课程组的随机组别槽位、每样本覆盖和每批组别组成均通过核验。",
              "- 选中候选的三个训练过程被正式实验逐轮重现；另一次完整重训的权重、先验、训练索引和预测也精确一致。",
              f"- 前两批 {preservation['files_checked']:,} 个文件、{preservation['bytes_checked']:,} 字节逐个哈希核验通过，没有改写。",
              "- 保存全部配置、环境、源码及哈希、逐步损失、轮次指标、评分、基线随机顺序与实际训练索引、每轮模型、最佳模型、预测、表格和图片。",
              "- 前两批测试结果已被查看，本批复用相同患者划分作连续开发比较；三个训练种子的结果不能代替新数据上的独立验证。", "",
              "## 重跑", "", "在项目根目录的PowerShell中执行：", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe run_batch3.py",
              f".\\.venv\\Scripts\\python.exe verify_results.py experiments\\{directory.name}",
              "```", "", "默认新建带时间戳的第三批目录；已写入实验方案的旧输出目录不允许复用。", ""]
    (directory / "batch3_report.md").write_text("\n".join(lines), encoding="utf-8")
    print({"report": str(directory / "batch3_report.md"), **conclusions}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    report(parser.parse_args().directory)
