"""Read-only diagnosis of a completed curriculum vs no-curriculum experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dataset import load_prepared
from fed import save_json


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(reference, output):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((reference / "protocol.json").read_text(encoding="utf-8"))
    data, _ = load_prepared(reference / "data" / "prepared.pt", "cpu")
    labels, masks = data["train"]["label"].numpy(), data["train"]["masks"].numpy()
    groups = labels.astype(int) * 4 + masks[:, 0].astype(int) + 2 * masks[:, 1].astype(int)
    exposures, ranks, summaries = [], [], []
    for seed in protocol["seeds"]:
        run = reference / f"fedcma_s{seed}"
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        for t in range(1, cfg["rounds"] + 1):
            with np.load(run / f"selection_round_{t:02d}.npz") as selection:
                for k in range(cfg["num_clients"]):
                    full, selected = selection[f"client_{k}_all"], selection[f"client_{k}_selected"]
                    draws = np.concatenate([selection[f"client_{k}_epoch_{e}_train"]
                                            for e in range(1, cfg["local_epochs"] + 1)])
                    for group in np.unique(groups[full]):
                        original = int((groups[full] == group).sum())
                        actual = int((groups[draws] == group).sum())
                        exposures.append({"seed": seed, "round": t, "client": k,
                            "label": int(group // 4), "mask_pattern": int(group % 4),
                            "original_unique": original, "selected_unique": int((groups[selected] == group).sum()),
                            "expected_draws": original * cfg["local_epochs"], "actual_draws": actual,
                            "draw_difference": actual - original * cfg["local_epochs"]})
        selected_rows = [r for r in exposures if r["seed"] == seed and r["label"] == 1]
        summaries.append({"seed": seed, "positive_draws_expected": sum(r["expected_draws"] for r in selected_rows),
                          "positive_draws_actual": sum(r["actual_draws"] for r in selected_rows),
                          "missing_positive_draws_expected": sum(r["expected_draws"] for r in selected_rows if r["mask_pattern"] != 3),
                          "missing_positive_draws_actual": sum(r["actual_draws"] for r in selected_rows if r["mask_pattern"] != 3)})
        for method in ("fedcma", "fedcma_wo_cl"):
            current = reference / f"{method}_s{seed}"
            result = json.loads((current / "result.json").read_text(encoding="utf-8"))
            with (current / "test_predictions.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            y = np.array([int(r["label"]) for r in rows])
            scores = np.array([float(r["probability"]) for r in rows])
            order = np.argsort(-scores, kind="stable")
            row = {"method": method, "seed": seed, "best_round": result["best_round"],
                   "validation_ap": result["best_validation"]["auprc"],
                   "test_auc": result["test_global"]["auc"], "test_ap": result["test_global"]["auprc"]}
            for top in (20, 50, 100, 200):
                row[f"positives_top_{top}"] = int(y[order[:top]].sum())
            row.update({f"ap_{key}": value["auprc"] for key, value in result["test_subgroups"].items()})
            ranks.append(row)
    write_csv(output / "batch2_exposure_by_group.csv", exposures)
    write_csv(output / "batch2_ranking_diagnostics.csv", ranks)
    save_json(output / "batch2_positive_exposure.json", summaries)
    lines = ["# 第三批调整前：第二批 AP 差异诊断", "",
             "以下均从第二批已保存的训练索引和预测读取，没有重新训练或修改第二批文件。", "",
             "## 可直接确认的现象", "",
             "| 种子 | 应有正例训练次数 | 实际正例训练次数 | 应有缺失正例训练次数 | 实际缺失正例训练次数 |",
             "|---|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['seed']} | {r['positive_draws_expected']} | {r['positive_draws_actual']} | {r['missing_positive_draws_expected']} | {r['missing_positive_draws_actual']} |")
    lines += ["", "| 种子 | 方法 | 测试 AP | 前50名正例 | 前100名正例 | 缺失样本 AP |",
              "|---|---|---:|---:|---:|---:|"]
    for r in ranks:
        lines.append(f"| {r['seed']} | {r['method']} | {r['test_ap']:.4f} | {r['positives_top_50']} | {r['positives_top_100']} | {r['ap_incomplete']:.4f} |")
    lines += ["", "## 机制解释与第三批针对性改动", "",
              "1. 第二批补齐了总训练次数，却没有补齐每类样本的训练次数。硬阈值把难正例尤其缺失正例排除，重复入选池不能恢复这些样本的曝光。上表给出了实际缺口。",
              "2. AP综合各召回水平的精确率，取决于正例在排名中的位置；AUROC改善不要求AP同时改善。第二批seed 44的高分段表现可以从前50/100名计数和分组AP进一步定位。指标定义见 [scikit-learn AP文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)。",
              "3. 原来的BCE归一化分母随评分小批次变化，同一绝对损失可能得到不同难度。第三批改为客户端内统一归一化，再在相同标签、相同缺失模式内部排序。",
              "4. 第三批保留所有样本，每个epoch每个样本恰好一次；课程只改变同组样本的顺序。保留基线随机排列的标签×缺失模式槽位，使每个小批次的组别计数也完全相同。",
              "5. 保留全样本预热，在末期恢复常规随机打乱；用小范围、仅验证集的对照选择课程强度和持续轮数，所有候选和失败结果都留档。",
              "正例曝光缺口是已确认事实；它是否是AP下降的主因仍是需要第三批干预检验的解释，不能仅凭相关现象确定因果。第三批仍复用前两批已查看过的测试集，属于连续开发比较。", ""]
    (output / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"positive_exposures": summaries, "ranking": ranks}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.reference, args.output)
