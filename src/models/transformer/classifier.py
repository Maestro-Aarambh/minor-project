"""
BaseClassifier wrapper for the multimodal transformer.

Why it exists:
The evaluation harness (evasive_eval) calls fit / predict_proba with a single
X matrix. The multimodal model consumes a PACKED matrix [static | byte seq]
(see src/data/multimodal_features.pack_views) and unpacks it internally, so
the transformer flows through the exact same harness as every baseline —
no special treatment in the metrics code.

Overfit/underfit controls:
- Stratified val split from the training pool only (never test/challenge).
- Static features standardized with mean/std computed on TRAIN ROWS ONLY;
  the scaler is saved in the checkpoint and reused at inference.
- Early stopping on val loss, best checkpoint restored.
- Dropout + AdamW weight decay + gradient clipping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data.multimodal_features import unpack_views
from src.models.base import BaseClassifier
from src.models.transformer import train as mm_train
from src.models.transformer.encoder import MultimodalTransformer


class TransformerClassifier(BaseClassifier):
    """Multimodal self-attention classifier over packed [static | bytes] rows."""

    def __init__(
        self,
        static_dim: int,
        seq_len: int = 4096,
        patch_size: int = 16,
        byte_embed_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.2,
        modality: str = "full",
        train_config: dict[str, Any] | None = None,
    ):
        self.static_dim = static_dim
        self.seq_len = seq_len
        self.hparams = {
            "static_dim": static_dim,
            "seq_len": seq_len,
            "patch_size": patch_size,
            "byte_embed_dim": byte_embed_dim,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "ff_dim": ff_dim,
            "dropout": dropout,
            "modality": modality,
        }
        self.train_config = train_config or {}
        self.model = MultimodalTransformer(**self.hparams)
        self.device = mm_train.get_device()
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        self._meta: dict[str, Any] = {}

    # ------------------------------------------------------------------ utils

    def _standardize(self, X_static: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None or self.scaler_std is None:
            raise RuntimeError("Scaler not fitted — call fit() or load() first.")
        return (X_static - self.scaler_mean) / self.scaler_std

    # ------------------------------------------------------------------- API

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransformerClassifier":
        cfg = {
            "seed": 42,
            "val_fraction": 0.1,
            "batch_size": 32,
            "epochs": 15,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "patience": 3,
            **self.train_config,
        }
        seed = int(cfg["seed"])
        self.device = mm_train.get_device(bool(cfg.get("use_gpu", True)))

        X_static, X_seq = unpack_views(X, self.static_dim)

        # Stratified val split from the training pool only (leakage-safe).
        idx_tr, idx_val = train_test_split(
            np.arange(len(y)),
            test_size=float(cfg["val_fraction"]),
            stratify=y,
            random_state=seed,
        )

        # Scaler statistics from TRAIN rows only (float64 avoids overflow on EMBER features).
        tr_static = X_static[idx_tr].astype(np.float64)
        mean = tr_static.mean(axis=0)
        std = tr_static.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.scaler_mean = mean.astype(np.float32)
        self.scaler_std = std.astype(np.float32)
        X_static_std = self._standardize(X_static).astype(np.float32)

        self._meta = mm_train.train_multimodal_classifier(
            self.model,
            (X_static_std[idx_tr], X_seq[idx_tr], y[idx_tr]),
            (X_static_std[idx_val], X_seq[idx_val], y[idx_val]),
            seed=seed,
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
        X_static, X_seq = unpack_views(X, self.static_dim)
        X_static_std = self._standardize(X_static).astype(np.float32)
        batch = int(self.train_config.get("predict_batch_size", 64))
        return mm_train.predict_proba_multimodal(
            self.model, X_static_std, X_seq, self.device, batch_size=batch
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "hparams": self.hparams,
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
                "meta": self._meta,
                "model_type": "multimodal_transformer",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TransformerClassifier":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        hparams = ckpt["hparams"]
        obj = cls(**hparams)
        obj.model.load_state_dict(ckpt["model_state"])
        obj.scaler_mean = ckpt.get("scaler_mean")
        obj.scaler_std = ckpt.get("scaler_std")
        obj._meta = ckpt.get("meta", {})
        obj.device = mm_train.get_device()
        obj.model.to(obj.device)
        return obj

    def get_params(self) -> dict[str, Any]:
        n_params = sum(p.numel() for p in self.model.parameters())
        return {**self.hparams, "n_parameters": int(n_params), **self._meta}
