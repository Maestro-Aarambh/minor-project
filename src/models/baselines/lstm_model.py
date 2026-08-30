"""
LSTM / BiLSTM baseline on histogram-derived byte sequences.

Why it exists:
LSTMs model sequential structure without attention — the direct comparison point
to ask whether self-attention adds value on evasive malware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseClassifier
from src.models.baselines import training


class ByteLSTM(nn.Module):
    """Embedding + BiLSTM + max-pool over time + linear head."""

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        pooled = out.max(dim=1).values
        pooled = self.dropout(pooled)
        return self.head(pooled)


class LSTMClassifier(BaseClassifier):
    """BiLSTM wrapper implementing BaseClassifier."""

    def __init__(
        self,
        seq_len: int = 4096,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        train_config: dict[str, Any] | None = None,
    ):
        self.seq_len = seq_len
        self.train_config = train_config or {}
        self.model = ByteLSTM(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.device = training.get_device()
        self._meta: dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMClassifier":
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
                "model_type": "lstm",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LSTMClassifier":
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
