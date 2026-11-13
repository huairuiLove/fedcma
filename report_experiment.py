"""Write the Chinese results report directly from verified machine-readable outputs."""
import argparse
import csv
import json
from pathlib import Path


def write_report(directory, report_path):
    directory = Path(directory)
    load = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    summary = load("summary.json")
    audit = load("verification.json")
    replay = load("retraining_verification.json")
    assert audit["all_passed"] and audit["n_runs"] == 9 and replay["passed"]
    by_method = {row["method"]: row for row in summary}
    with (directory / "per_seed.csv").open(encoding="utf-8") as stream:
        per_seed = list(csv.DictReader(stream))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*/result.json")]
    total_seconds = sum(r["elapsed_seconds"] for r in runs)
    peak = max(r["peak_cuda_allocated_mb"] for r in runs)
    steps = sum(r["step_records"] for r in audit["runs"])
    names = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedcma": "FedCMA 本文方法"}
    table = ["| 方法 | 测试AUROC | 测试AP |", "|---|---:|---:|"]
    for r in summary:
        table.append(f"| {names[r['method']]} | {r['auc_mean']:.4f} ± {r['auc_std']:.4f} | "
                     f"{r['auprc_mean']:.4f} ± {r['auprc_std']:.4f} |")
    seed_table = ["| 方法 | 种子 | 验证选定轮次 | 测试AUROC | 测试AP |", "|---|---:|---:|---:|---:|"]
    for r in per_seed:
        seed_table.append(f"| {names[r['method']]} | {r['seed']} | {r['best_round']} | "
                          f"{float(r['auc']):.4f} | {float(r['auprc']):.4f} |")
    ours = by_method["fedcma"]
    gains = []
    for key in ("fedavg", "fedprox"):
        base = by_method[key]
        gains.append(f"相对{names[key]}，平均AUROC提高{100*(ours['auc_mean']-base['auc_mean']):.2f}个百分点，"
                     f"平均AP提高{100*(ours['auprc_mean']-base['auprc_mean']):.2f}个百分点。")
    paragraph = (
        "在一个具体实施例中，使用现有MIMIC-IV生理时序与MIMIC-CXR放射报告处理结果开展小规模实验。"
        "任务为根据48小时输入窗口预测院内死亡。训练子集包含3072条样本，验证集427条、测试集1131条，"
        "训练、验证与测试患者无交集。将训练患者按标签相关的狄利克雷分布（α=0.5）分配至5个模拟客户端，"
        "在约50%的样本中移除一个模态，对应约25%的模态条目缺失。采用轻量时序卷积编码器与冻结文本嵌入投影，"
        "通过双模态自注意力进行融合。本文方法为每个客户端维护128维模态先验，并采用128→64→128的先验生成器。"
        "三种方法均使用Adam优化器、学习率0.001、批量大小64，进行5轮全局通信，每轮本地训练3个epoch；"
        "本文方法按动态难度阈值选择本轮参与训练的样本。FedProx近端项系数为0.01。"
        "本文方法的特征对齐与先验正则权重均为0.001，难度混合系数为0.7，难度阈值从0.6增至1，"
        "本地与全局先验混合系数从0.2增至0.6。本文方法在预先列明的5组专用参数组合中按验证AP选择配置，"
        "FedAvg和FedProx使用上述共同固定配置。固定数据划分和缺失掩码，分别采用42、43、44三个训练随机种子，"
        "每次以验证集AP最高的轮次选择模型，再在测试集评估。"
        f"本文方法的测试AUROC为{ours['auc_mean']:.4f}±{ours['auc_std']:.4f}，"
        f"AP为{ours['auprc_mean']:.4f}±{ours['auprc_std']:.4f}；"
        f"FedAvg分别为{by_method['fedavg']['auc_mean']:.4f}±{by_method['fedavg']['auc_std']:.4f}和"
        f"{by_method['fedavg']['auprc_mean']:.4f}±{by_method['fedavg']['auprc_std']:.4f}；"
        f"FedProx分别为{by_method['fedprox']['auc_mean']:.4f}±{by_method['fedprox']['auc_std']:.4f}和"
        f"{by_method['fedprox']['auprc_mean']:.4f}±{by_method['fedprox']['auprc_std']:.4f}。"
        "上述数值为三个训练种子的均值与样本标准差，AP采用average precision作为AUPRC评价指标。"
        "在本次小规模实验和固定基线配置下，本文方法的平均AUROC与AP均高于两个基线。"
        "本实验未进行消融，不使用原稿的其他数据集结果或旧提升比例。")
    (directory / "patent_experiment_paragraph.txt").write_text(paragraph + "\n", encoding="utf-8")
    report = f"""# 本地专利实验结果

2026年11月13日已完成三种方法、三个种子共9次正式运行。FedCMA在当前固定数据划分和基线配置下的平均测试AUROC与AP均最高。数字直接来自本次保存的预测与结果文件，不使用旧稿的实验数字。

## 测试集结果

{chr(10).join(table)}

均值±样本标准差，三个种子全部计入。主要指标来自同一测试集1131条样本的合并预测，包含162例阳性。AP为average precision，并非梯形积分PR面积。

{' '.join(gains)}

## 每次运行结果

{chr(10).join(seed_table)}

仅验证AP用于选轮次；测试结果没有参与配置选择或轮次选择。FedCMA额外做了5种专用参数组合的验证搜索，基线使用共同固定配置。该调参预算差异已保留在`validation_search_20261113/search_plan.json`和完整搜索结果中。

## 实验设置与硬件

训练3072条、验证427条、测试1131条；按患者检查无跨集合重叠。5个模拟客户端，α=0.5；约50%的样本缺失一个模态。三种方法使用相同小型骨干、数据、初始权重规则、Adam学习率0.001、批量64、5轮通信及每轮3个本地epoch。本文方法使用动态筛选，所以实际处理样本数与两个基线不完全相同。

本机为RTX 5090 D v2（24 GB）、约32 GB内存，使用Python 3.12和PyTorch 2.14.1+cu130。本文模型338306个可训练参数，基线305154个；另有每客户端两个128维先验向量。冻结文本模型已经转换成输入嵌入，训练时不用加载大型语言模型。

正式9次运行记录的训练及评价时间累计约{total_seconds:.1f}秒，不包括依赖安装、数据准备和文档生成。首次CUDA初始化影响首个运行时间，不将这些时间用于方法速度优劣结论。PyTorch记录的峰值已分配显存为{peak:.1f} MiB，不含CUDA上下文和桌面显示占用。

## 数据与复核记录

- 保存{steps}条逐批训练记录，包括任务、对齐、先验正则、近端和总损失，以及梯度范数与学习率。
- 保存9次运行共45个逐轮模型及9份最佳模型，每个检查点包含全局权重、本地先验和全局参考先验。
- 保存逐轮课程难度、入选样本索引、验证预测，以及每次运行的最终测试预测。
- 保存原始输入和准备数据的SHA256、训练抽样与患者分配、归一化参数、源码快照、完整配置和软件环境。
- 全部9份最佳模型重新加载后，逐样本预测最大差异均为0，指标从CSV复算通过。
- 另将FedCMA种子42从头重新训练，41个模型张量、本地先验、参考先验与测试预测CSV完全一致。

数据的两个通道在当前训练集内为常量（毛细血管再充盈、GCS总分）；当前预处理将其规范为零，不虚构变化。输入来源是已有处理文件；本次没有重新从原始临床数据库抽取数据，也不把5个模拟客户端描述为5家真实医院。

三个训练种子的标准差仅反映固定数据划分下的训练变动。这里没有消融结果，也没有针对跨机构推广或统计显著性的实验结论。

## 专利实验段落替换参考

{paragraph}

这段文字也保存为`experiments/final_ihm_20261113/patent_experiment_paragraph.txt`。原Word文件保持原样，便于与老师确认后替换。

## 文件与复现入口

- [原始结果表](experiments/final_ihm_20261113/summary.csv)
- [全部种子明细](experiments/final_ihm_20261113/per_seed.csv)
- [训练曲线](experiments/final_ihm_20261113/training_curves.png)
- [测试结果图](experiments/final_ihm_20261113/test_comparison.png)
- [全套复核记录](experiments/final_ihm_20261113/verification.json)
- [重新训练复核](experiments/final_ihm_20261113/retraining_verification.json)
- [技术对照](IMPLEMENTATION_REVIEW.md)
- [运行说明](README.md)

在项目目录执行`./run_local.ps1`即可按固定配置重新运行，输出到新的时间戳目录。原始结果、开发试跑、全部验证搜索和额外复现运行都保存在`experiments/`，没有覆盖或筛除不理想的实验。
"""
    Path(report_path).write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("EXPERIMENT_REPORT.md"))
    args = parser.parse_args()
    write_report(args.directory, args.output)
