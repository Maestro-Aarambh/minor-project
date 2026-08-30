"""
Shared PyTorch training loop for sequence baselines (CNN, LSTM).

Why it exists:
CNN and LSTM share the same training contract (val split, BCE loss, early
stopping, seeding) so the comparison stays fair and the code stays DRY.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_binary_classifier(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    batch_size: int = 128,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 3,
    device: torch.device | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """
    Train a PyTorch binary classifier on int64 byte sequences (N, seq_len).

    Returns training metadata (best epoch, val loss).
    """
    device = device or get_device()
    set_seed(seed)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.long),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_val, dtype=torch.long),
            torch.as_tensor(y_val, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
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
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(-1)
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
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Return P(malicious) for each row."""
    model.eval()
    model = model.to(device)
    probs: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.long, device=device)
        logits = model(xb).squeeze(-1)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)
