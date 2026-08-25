"""
Standard binary classification metrics for malware detection.

Why it exists:
Accuracy alone is misleading under class imbalance. The paper reports the full
suite (precision, recall, F1, ROC-AUC, PR-AUC) so readers can judge trade-offs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int32)

    out: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "n_malicious": int(np.sum(y_true == 1)),
        "n_benign": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC-AUC / PR-AUC need both classes present.
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        out["pr_auc"] = float(auc(recall, precision))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        # Detection rate at 1% FPR — common malware-detection operating point.
        target_fpr = 0.01
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
        out["tpr_at_fpr_1pct"] = float(tpr[idx])
        out["fpr_at_report_point"] = float(fpr[idx])
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        out["tpr_at_fpr_1pct"] = float("nan")
        out["fpr_at_report_point"] = float("nan")
        # Detection rate among all-malicious challenge samples.
        out["detection_rate"] = float(np.mean(y_pred == 1))

    return out
