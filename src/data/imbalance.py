"""
Class-imbalance helpers for binary malware detection.

Why it exists:
EMBER splits are roughly balanced weekly, but any stratified subsample and the
challenge set (all malicious) change class ratios. Explicit class weights keep
accuracy from being a misleading headline metric.
"""

from __future__ import annotations

import numpy as np


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Inverse-frequency weights for labels {0, 1}."""
    y = np.asarray(y).astype(np.int32)
    n = len(y)
    counts = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1))}
    weights = {}
    for cls, count in counts.items():
        weights[cls] = (n / (2.0 * count)) if count > 0 else 0.0
    return weights


def sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights matching class_weights(y)."""
    cw = class_weights(y)
    return np.array([cw[int(label)] for label in y], dtype=np.float32)
