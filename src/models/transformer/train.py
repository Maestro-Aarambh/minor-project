"""
Training loop for the multimodal transformer (two-input version of
src/models/baselines/training.py).

Why it exists:
The baseline loop feeds one tensor per batch; the multimodal model needs
(static, sequence) pairs. Everything else — validation split from the TRAINING
POOL ONLY, class-weighted BCE, early stopping on val loss, best-checkpoint
restore — mirrors the baseline loop so the comparison stays fair and the
overfitting controls are identical.

Leakage note: the val split is carved from training rows only (stratified,
seeded). The official EMBER test set and the challenge set never enter fit().
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.baselines.training import get_device, set_seed  # noqa: F401  (re-export)


def _make_loader(
    X_static: np.ndarray,
    X_seq: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.as_tensor(X_static, dtype=torch.float32),
            torch.as_tensor(X_seq, dtype=torch.long),
            torch.as_tensor(y, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_multimodal_classifier(
    model: nn.Module,
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    seed: int = 42,
    batch_size: int = 32,
    epochs: int = 15,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 3,
    device: torch.device | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """
    Train on pre-split (train, val) triples: (X_static, X_seq, y).

    The caller performs the stratified split and static standardization so that
    scaler statistics come from training rows only.
    """
    device = device or get_device()
    set_seed(seed)

    Xs_tr, Xq_tr, y_tr = train_data
    Xs_val, Xq_val, y_val = val_data

    train_loader = _make_loader(Xs_tr, Xq_tr, y_tr, batch_size, shuffle=True)
    val_loader = _make_loader(Xs_val, Xq_val, y_val, batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    if pos > 0 and neg > 0:
        pos_weight = torch.tensor([neg / pos], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xs, xq, yb in train_loader:
            xs, xq, yb = xs.to(device), xq.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xs, xq).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for xs, xq, yb in val_loader:
                xs, xq, yb = xs.to(device), xq.to(device), yb.to(device)
                logits = model(xs, xq).squeeze(-1)
                val_loss += float(criterion(logits, yb).item())
                val_batches += 1
        val_loss /= max(val_batches, 1)
        if progress:
            progress(
                f"[train] epoch {epoch}/{epochs} train_loss={train_loss/max(n_batches,1):.4f} "
                f"val_loss={val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                if progress:
                    progress(f"[train] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "device": str(device),
    }


@torch.no_grad()
def predict_proba_multimodal(
    model: nn.Module,
    X_static: np.ndarray,
    X_seq: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return P(malicious) per row, batched to stay inside 4 GB VRAM."""
    model.eval()
    model = model.to(device)
    probs: list[np.ndarray] = []
    for start in range(0, len(X_seq), batch_size):
        xs = torch.as_tensor(
            X_static[start : start + batch_size], dtype=torch.float32, device=device
        )
        xq = torch.as_tensor(X_seq[start : start + batch_size], dtype=torch.long, device=device)
        logits = model(xs, xq).squeeze(-1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)
