"""Diagnose saved batch 3 ordering and validation/test disagreement read-only."""
import argparse
from pathlib import Path

import numpy as np

from fed import save_json
from report_batch2 import read, write_csv


def analyze(reference, output):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = read(reference / "protocol.json")
    movement, comparisons = [], []
    for seed in protocol["seeds"]:
        full, control = reference / f"fedcma_s{seed}", reference / f"fedcma_wo_cl_s{seed}"
        cfg, history = read(full / "config.json"), read(full / "history.json")
        control_history = read(control / "history.json")
        a, b = read(full / "result.json"), read(control / "result.json")
        comparisons.append({"seed": seed, "validation_best_ap_gain": a["best_validation"]["auprc"] - b["best_validation"]["auprc"],
            "validation_last2_ap_gain": float(np.mean([r["val"]["auprc"] for r in history[-2:]]) - np.mean([r["val"]["auprc"] for r in control_history[-2:]])),
            "test_ap_gain": a["test_global"]["auprc"] - b["test_global"]["auprc"],
            "full_best_round": a["best_round"], "no_cl_best_round": b["best_round"]})
        for r in history:
            positions, changed, changed_batch = 0, 0, 0
            with np.load(full / f"selection_round_{r['round']:02d}.npz") as selection:
                for k in range(cfg["num_clients"]):
                    for e in range(1, cfg["local_epochs"] + 1):
                        base, actual = selection[f"client_{k}_epoch_{e}_random"], selection[f"client_{k}_epoch_{e}_train"]
                        lookup = np.full(int(base.max()) + 1, -1, dtype=int)
                        batch_ids = np.arange(len(base)) // cfg["batch_size"]
                        lookup[base] = batch_ids
                        positions += len(base)
                        changed += int((base != actual).sum())
                        changed_batch += int((lookup[actual] != batch_ids).sum())
            movement.append({"seed": seed, "round": r["round"], "phase": r["curriculum"]["phase"],
                "strength": r["curriculum"].get("order_strength", 0.), "positions": positions,
                "changed_positions": changed, "changed_batches": changed_batch,
                "changed_position_fraction": changed / positions, "changed_batch_fraction": changed_batch / positions})
    write_csv(output / "batch3_order_movement.csv", movement)
    write_csv(output / "batch3_validation_test_gaps.csv", comparisons)
    active = [r for r in movement if r["strength"] > 0]
    summary = {"active_changed_position_fraction": sum(r["changed_positions"] for r in active) / sum(r["positions"] for r in active),
               "active_changed_batch_fraction": sum(r["changed_batches"] for r in active) / sum(r["positions"] for r in active),
               "validation_test_gaps": comparisons}
    save_json(output / "diagnosis_summary.json", summary)
    lines = ["# 第四批调整前：第三批课程机制诊断", "",
        "本文件只读取第三批已保存的结果和训练索引。", "",
        f"第三批两个课程轮中，虽然组别计数一致，但实际有 {summary['active_changed_position_fraction']:.2%} 的样本处理位置改变，{summary['active_changed_batch_fraction']:.2%} 的样本进入了不同的小批次。排序强度系数小，并不等于训练序列改动小。",
        "第三批难度分数在每个客户端开始本轮训练时计算一次，随后三个本地epoch复用；这是源码可确认的行为。该分数在本地模型更新后可能不再反映当前难度，这一影响尚未被单独量化。", "",
        "| 种子 | 最佳验证 AP 增益 | 最后两轮验证 AP 平均增益 | 测试 AP 增益 |", "|---|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['seed']} | {r['validation_best_ap_gain']:+.4f} | {r['validation_last2_ap_gain']:+.4f} | {r['test_ap_gain']:+.4f} |")
    lines += ["", "验证最高点的提升没有稳定对应到测试AP提升。第四批选参数同时看最佳验证AP和最后两轮平均验证AP，避免仅凭一次较高的验证值。", "",
        "第四批保持原随机顺序，保留所有样本，依据当前训练小批次的BCE在同标签、同缺失模式内的难度名次赋予有界权重。预热后逐轮增加难例权重强度；每个组的权重和保持为该组样本数。这样可以保持样本次数、批次组成和类别/缺失模式的总权重。",
        "逐步重视困难样本的设计参考了 [Focal Loss论文](https://arxiv.org/abs/1708.02002) 对易例主导问题的处理思路；本批实现为有界、组内归一的名次权重，不是该论文的原始Focal Loss公式。它对当前任务的作用由本批实验检验。",
        "AP基于精确率与召回率，定义见 [scikit-learn文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)。当前BCE难度不等同于样本对AP的真实训练价值，因此上述调整是待检验的机制假设。", ""]
    (output / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    analyze(args.reference, args.output)
