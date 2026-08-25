"""
Standard-set vs evasive-challenge evaluation and drop-off table.

Why it exists:
This is the central experiment of the project. Raw accuracy on the standard
test set is not the contribution — the contribution is how much performance
*drops* on samples that initially evaded every AV engine on VirusTotal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.metrics import binary_metrics
from src.models.base import BaseClassifier


def evaluate_standard_and_evasive(
    model: BaseClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_challenge: np.ndarray,
    y_challenge: np.ndarray,
    *,
    threshold: float = 0.5,
    mix_test_benign_into_challenge: bool = True,
) -> dict[str, Any]:
    """
    Evaluate once on the temporal test set and once on the evasive challenge set.

    Following the official EMBER2024 eval script, the challenge ROC/PR curves are
    computed after concatenating test-set benign files with the (all-malicious)
    challenge samples — otherwise ROC-AUC is undefined on an all-positive set.
    We also report raw challenge detection rate on challenge-only samples.
    """
    test_scores = model.predict_proba(X_test)
    test_metrics = binary_metrics(y_test, test_scores, threshold=threshold)

    challenge_only_scores = model.predict_proba(X_challenge)
    challenge_only_pred = (challenge_only_scores >= threshold).astype(np.int32)
    challenge_detection_rate = float(np.mean(challenge_only_pred == 1))

    if mix_test_benign_into_challenge:
        X_benign = X_test[y_test == 0]
        y_benign = y_test[y_test == 0]
        X_chal_eval = np.concatenate([X_benign, X_challenge], axis=0)
        y_chal_eval = np.concatenate([y_benign, y_challenge], axis=0)
    else:
        X_chal_eval, y_chal_eval = X_challenge, y_challenge

    challenge_scores = model.predict_proba(X_chal_eval)
    challenge_metrics = binary_metrics(y_chal_eval, challenge_scores, threshold=threshold)
    challenge_metrics["challenge_only_detection_rate"] = challenge_detection_rate
    challenge_metrics["challenge_only_n"] = int(len(y_challenge))

    dropoff = {}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tpr_at_fpr_1pct"):
        a = test_metrics.get(key)
        b = challenge_metrics.get(key)
        if a is None or b is None or np.isnan(a) or np.isnan(b):
            dropoff[key] = float("nan")
        else:
            dropoff[key] = float(a - b)

    return {
        "standard_test": test_metrics,
        "evasive_challenge": challenge_metrics,
        "dropoff_standard_minus_evasive": dropoff,
    }
