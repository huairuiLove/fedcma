"""Generate a complete ablation table and optional patent paragraph from real runs."""
import argparse
import csv
import json
from pathlib import Path

NAMES = {"fedcma": "完整FedCMA", "fedcma_wo_prior": "去掉先验模块",
         "fedcma_wo_cl": "去掉课程学习", "fedcma_wo_both": "两者都去掉"}


def report(directory, output):
    directory, output = Path(directory), Path(output)
    load = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    protocol, verification, controls = (load(n) for n in ("protocol.json", "verification.json", "control_verification.json"))
    assert verification["all_passed"] and controls["all_passed"]
    summary = {r["method"]: r for r in load("summary.json")}
    full = summary["fedcma"]
    methods = protocol["methods"]
    results = {(r["method"], r["seed"]): r for r in load("completed_runs.json")}
    table = ["| 模型 | 先验模块 | 课程学习 | 测试AUROC | 测试AP |", "|---|---|---|---:|---:|"]
    seed_table = ["| 模型 | 种子 | 验证选中轮次 | 测试AUROC | 测试AP |", "|---|---:|---:|---:|---:|"]
    comparisons, descriptions = [], []
    for method in methods:
        row, flags = summary[method], protocol["components"][method]
        table.append(f"| {NAMES[method]} | {'有' if flags['prior'] else '无'} | {'有' if flags['curriculum'] else '无'} | "
                     f"{row['auc_mean']:.4f} ± {row['auc_std']:.4f} | {row['auprc_mean']:.4f} ± {row['auprc_std']:.4f} |")
        for seed in protocol["seeds"]:
            r = results[method, seed]
            seed_table.append(f"| {NAMES[method]} | {seed} | {r['best_round']} | "
                              f"{r['test_global']['auc']:.4f} | {r['test_global']['auprc']:.4f} |")
        if method == "fedcma":
            continue
        record = {"variant": method}
        clauses = []
        for metric, name in (("auc", "AUROC"), ("auprc", "AP")):
            delta = (full[metric + "_mean"] - row[metric + "_mean"]) * 100
            per_seed_delta = [results["fedcma", s]["test_global"][metric] - results[method, s]["test_global"][metric]
                              for s in protocol["seeds"]]
            record[metric + "_full_minus_variant_pp"] = delta
            record[metric + "_positive_seed_count"] = sum(x > 0 for x in per_seed_delta)
            clauses.append(f"{name}{'高' if delta >= 0 else '低'}{abs(delta):.2f}个百分点")
        comparisons.append(record)
        descriptions.append(f"相对“{NAMES[method]}”，完整模型的平均" + "、".join(clauses) + "。")
    with (directory / "paired_differences.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparisons[0].keys())
        writer.writeheader()
        writer.writerows(comparisons)
    all_higher = all(r[metric + "_full_minus_variant_pp"] > 0 for r in comparisons for metric in ("auc", "auprc"))
    conclusion = ("完整模型的平均AUROC和AP均高于三个消融版本，可作为当前实验设置下组合方案有效的补充结果。"
                  if all_higher else "消融结果对不同指标或模块的支持并不完全一致，应按完整表格描述，不能写成所有模块在所有指标上都提高效果。")
    if (full["auc_mean"] > summary["fedcma_wo_prior"]["auc_mean"]
            and full["auprc_mean"] > summary["fedcma_wo_prior"]["auprc_mean"]
            and summary["fedcma_wo_cl"]["auprc_mean"] > full["auprc_mean"]):
        conclusion = ("本次结果支持先验模块的作用；课程学习没有显示一致增益，去掉课程学习的版本获得了更高的平均AP。"
                      "完整模型与去掉课程版本的平均AUROC相近。这组消融可保留作补充分析，不适合作为两个模块都改善所有指标的证据。")
    paragraph = ("进一步在相同数据划分、缺失掩码、训练预算和三个随机种子下进行模块消融。"
                 "保持已有完整模型的参数设置，分别移除先验模块、课程学习，以及同时移除两者；不对消融版本重新调参。"
                 "移除先验模块包括停用可学习先验、生成器、特征对齐、先验正则与先验共享，并对缺失模态使用零特征；"
                 "移除课程学习时，每轮使用全部本地训练样本。仍按验证集AP选择检查点，然后计算测试指标。")
    for method in methods:
        row = summary[method]
        paragraph += (f"{NAMES[method]}的测试AUROC为{row['auc_mean']:.4f}±{row['auc_std']:.4f}，"
                      f"AP为{row['auprc_mean']:.4f}±{row['auprc_std']:.4f}；")
    paragraph = paragraph.rstrip("；") + "。数值为三个种子的均值与样本标准差。" + conclusion
    (directory / "patent_ablation_paragraph.txt").write_text(paragraph + "\n", encoding="utf-8")
    n_steps = sum(r["step_records"] for r in verification["runs"])
    n_runs = verification["n_runs"]
    seconds = sum(r["elapsed_seconds"] for r in results.values())
    text = f"""# FedCMA补充消融实验

已完成完整模型及三个消融版本，各使用种子42、43、44，共{n_runs}次运行。{conclusion}

## 测试结果

{chr(10).join(table)}

全部种子均保留，数值为均值±样本标准差。AP采用average precision，与主实验口径一致。

{' '.join(descriptions)}

## 模块定义与比较范围

本次以已完成的主实验配置为固定起点，使用相同的3072条训练、427条验证、1131条测试样本，以及同一患者划分、缺失掩码、5个客户端、5轮通信、每轮3个本地epoch、学习率0.001。配置文件和数据直接从主实验复制，不重新抽样或搜索超参数。

“先验模块”指完整功能组合：可学习先验、128→64→128生成器、特征对齐损失、先验正则化和跨客户端先验共享。去掉该模块后用零特征填补缺失模态，但可保留课程筛选。因而这项消融不能单独归因于先验向量中的某个参数或某一项损失。

去掉课程学习时保留完整先验功能，每轮使用全部本地样本。各组本地epoch相同，但实际处理样本数可不同；实际选样和步数均有记录。“两者都去掉”在当前实现下等价于FedAvg。

所有模型仍以验证AP选轮次，测试结果不用于调整参数。本次配置是在完整模型的已有验证搜索后固定的，消融版本没有单独调优。三个种子共享同一数据划分，结果描述限于此设置，不作统计显著性断言。

## 每个种子的结果

{chr(10).join(seed_table)}

## 复现与完整性核验

12份最佳模型均已重放，逐样本预测最大差异为0，指标从预测CSV重新计算通过。另验证了6组控制比较：3个完整模型与原主实验的同种子模型完全一致；3个“两者都去掉”模型与原FedAvg完全一致，包括权重、先验存档和测试预测。

记录{n_steps}条逐批损失、60个逐轮检查点、12个最佳检查点、每轮选样、源码、环境与配置。运行记录中的训练和评价时间累计约{seconds:.1f}秒，包含首个运行的CUDA初始化，不含环境配置与额外重放检查。

## 是否用于专利

这组数据可单独作为补充材料，主比较表仍可独立使用。若加入消融部分，可使用以下事实描述；是否纳入正文由你和老师决定。

{paragraph}

## 文件与命令

- 完整结果目录：`{directory.resolve()}`
- `summary.csv`：完整四组均值与标准差
- `per_seed.csv`：全部种子及选中轮次
- `paired_differences.csv`：完整模型与各消融版本的逐项差异
- `verification.json`：12次模型重放与指标复算
- `control_verification.json`：与原主实验的6组等价检查
- `training_curves.png`、`test_comparison.png`：训练曲线和测试对比图
- `patent_ablation_paragraph.txt`：可选的专利补充文字

在项目目录运行`.\\.venv\\Scripts\\python.exe run_ablations.py`，会创建新的时间戳目录并复跑同一设计。复核已有结果可运行`.\\.venv\\Scripts\\python.exe verify_results.py experiments\\ablation_ihm_20261113`。
"""
    output.write_text(text, encoding="utf-8")
    (directory / "ablation_report.md").write_text(text, encoding="utf-8")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ABLATION_REPORT.md"))
    args = parser.parse_args()
    report(args.directory, args.output)
