"""Pooled binary metrics; AUPRC is explicitly average precision (AP)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss


def compute_metrics(y_true, y_score):
    y, score = np.asarray(y_true).astype(int), np.asarray(y_score).astype(float)
    if not len(y) or not np.isfinite(score).all():
        raise ValueError("Empty evaluation or non-finite predictions")
    return {"auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
            "auprc": float(average_precision_score(y, score)) if y.sum() else None,
            "bce": float(log_loss(y, np.clip(score, 1e-7, 1 - 1e-7), labels=[0, 1])),
            "n": len(y), "positives": int(y.sum())}
