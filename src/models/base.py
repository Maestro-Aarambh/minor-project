"""
Common classifier interface shared by all models in this project.

Why it exists:
Every baseline and the transformer must expose the same API so the evaluation
harness treats them identically. That makes the paper's "fair comparison"
claim defensible — no model gets special treatment in the metrics code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseClassifier(ABC):
    """Minimal train / predict / persist contract for all classifiers."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        """Train on feature matrix X and binary labels y (0=benign, 1=malicious)."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(malicious) as a 1-D array of shape (n_samples,)."""

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseClassifier":
        """Load a previously saved model."""

    def get_params(self) -> dict[str, Any]:
        return {}
