"""
1D CNN baseline on histogram-derived byte sequences.

Why it exists:
CNNs test whether *local* byte patterns are enough for detection. This is the
standard foil to LSTM (sequential) and transformer (attention) in malware papers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseClassifier
from src.models.baselines import training


class ByteCNN(nn.Module):
    """Embedding + stacked 1D convolutions + global max pool + linear head."""

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 64,
        num_filters: tuple[int, ...] = (128, 256, 256),
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        layers: list[nn.Module] = []
        in_ch = embed_dim
        for out_ch in num_filters:
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(num_filters[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) long
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed, seq)
        feat = self.conv(emb)
        pooled = feat.max(dim=2).values
        return self.head(pooled)


class CNNClassifier(BaseClassifier):
    """1D CNN wrapper implementing BaseClassifier."""

    def __init__(
        self,
        seq_len: int = 4096,
        embed_dim: int = 64,
        num_filters: tuple[int, ...] = (128, 256, 256),
        kernel_size: int = 7,
        dropout: float = 0.2,
        train_config: dict[str, Any] | None = None,
    ):
        self.seq_len = seq_len
        self.train_config = train_config or {}
        self.model = ByteCNN(
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.device = training.get_device()
        self._meta: dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNClassifier":
        cfg = {
            "seed": 42,
            "val_fraction": 0.1,
            "batch_size": 128,
            "epochs": 10,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "patience": 3,
            **self.train_config,
        }
        self.device = training.get_device(bool(cfg.get("use_gpu", True)))
        self._meta = training.train_binary_classifier(
            self.model,
            X,
            y,
            seed=int(cfg["seed"]),
            val_fraction=float(cfg["val_fraction"]),
            batch_size=int(cfg["batch_size"]),
            epochs=int(cfg["epochs"]),
            learning_rate=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
            patience=int(cfg["patience"]),
            device=self.device,
        )
        self._meta["train_config"] = cfg
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return training.predict_proba(self.model, X, self.device)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "meta": self._meta,
                "seq_len": self.seq_len,
                "model_type": "cnn",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CNNClassifier":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(seq_len=int(ckpt.get("seq_len", 4096)))
        obj.model.load_state_dict(ckpt["model_state"])
        obj._meta = ckpt.get("meta", {})
        obj.device = training.get_device()
        obj.model.to(obj.device)
        return obj

    def get_params(self) -> dict[str, Any]:
        return {"seq_len": self.seq_len, **self._meta}
