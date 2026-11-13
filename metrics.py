"""Metrics: IHM -> AUC / AUPRC (positive class); Pheno -> micro/macro AP and AUC."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(y_true, y_score, task):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if task == "ihm":
        return {"auc": float(roc_auc_score(y_true, y_score)),
                "auprc": float(average_precision_score(y_true, y_score))}
    m = {}
    for agg in ["micro", "macro"]:
        m[f"{agg}_ap"] = float(average_precision_score(y_true, y_score, average=agg))
        m[f"{agg}_auc"] = float(roc_auc_score(y_true, y_score, average=agg))
    return m


def weighted_average(per_client, weights):
    """per_client: list of metric dicts; weights: list of n_test."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    keys = per_client[0].keys()
    return {k: float(sum(wi * m[k] for wi, m in zip(w, per_client))) for k in keys}
