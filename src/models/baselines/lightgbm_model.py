"""
LightGBM baseline on EMBER static features.

Why it exists:
LightGBM on engineered PE features is the strongest classical baseline published
on EMBER-style datasets. It establishes the performance floor the transformer
must beat, especially on the evasive challenge set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from src.models.base import BaseClassifier


DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "boosting": "gbdt",
    "num_iterations": 500,
    "learning_rate": 0.1,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "metric": ["auc", "binary_logloss"],
    "verbosity": -1,
    "seed": 42,
}


class LightGBMClassifier(BaseClassifier):
    """Binary LightGBM wrapper implementing BaseClassifier."""

    def __init__(self, params: dict[str, Any] | None = None, val_fraction: float = 0.1):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.val_fraction = val_fraction
        self.model: lgb.Booster | None = None
        self.best_iteration: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMClassifier":
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.val_fraction,
            stratify=y,
            random_state=int(self.params.get("seed", 42)),
        )
        # Categorical indices from EMBER feature v3 / thrember examples.
        cat_features = [2, 3, 4, 5, 6, 701, 702]
        train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=cat_features)
        self.model = lgb.train(
            self.params,
            train_set,
            valid_sets=[val_set],
            valid_names=["val"],
        )
        self.best_iteration = self.model.best_iteration or self.params.get("num_iterations")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        kwargs = {}
        if self.best_iteration:
            kwargs["num_iteration"] = self.best_iteration
        return self.model.predict(X, **kwargs)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path), num_iteration=self.best_iteration)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMClassifier":
        obj = cls()
        obj.model = lgb.Booster(model_file=str(path))
        obj.best_iteration = obj.model.best_iteration
        return obj

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)
