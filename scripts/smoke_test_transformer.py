"""
Smoke test for the multimodal transformer on synthetic data (CPU, seconds).

Verifies: pack/unpack round-trip, scaler fit on train rows only, training loop
with early stopping, batched predict_proba, save/load round-trip, and the
evasive_eval harness end-to-end. No real dataset or GPU required.

Run: python scripts/smoke_test_transformer.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.multimodal_features import pack_views, unpack_views
from src.evaluation.evasive_eval import evaluate_standard_and_evasive
from src.models.transformer.classifier import TransformerClassifier

STATIC_DIM = 64
SEQ_LEN = 256  # small for speed; patch_size 16 -> 16 tokens
N_TRAIN, N_TEST, N_CHAL = 200, 80, 20

rng = np.random.default_rng(0)


def make_split(n: int, all_malicious: bool = False):
    y = np.ones(n, dtype=np.int32) if all_malicious else rng.integers(0, 2, n).astype(np.int32)
    # Give the classes a learnable signal in both views.
    X_static = rng.normal(0, 1, (n, STATIC_DIM)).astype(np.float32) + y[:, None] * 0.8
    X_seq = rng.integers(0, 256, (n, SEQ_LEN)).astype(np.int64)
    X_seq[y == 1, :32] = 200  # malicious byte motif
    return X_static, X_seq, y


Xs_tr, Xq_tr, y_tr = make_split(N_TRAIN)
Xs_te, Xq_te, y_te = make_split(N_TEST)
Xs_ch, Xq_ch, y_ch = make_split(N_CHAL, all_malicious=True)

# --- pack/unpack round-trip -------------------------------------------------
X_tr = pack_views(Xs_tr, Xq_tr)
a, b = unpack_views(X_tr, STATIC_DIM)
assert np.allclose(a, Xs_tr) and np.array_equal(b, Xq_tr), "pack/unpack mismatch"
print("[smoke] pack/unpack OK")

# --- fit (tiny model, 3 epochs) ----------------------------------------------
model = TransformerClassifier(
    static_dim=STATIC_DIM,
    seq_len=SEQ_LEN,
    patch_size=16,
    byte_embed_dim=8,
    d_model=32,
    nhead=2,
    num_layers=1,
    ff_dim=64,
    dropout=0.1,
    train_config={"epochs": 3, "batch_size": 32, "use_gpu": False, "predict_batch_size": 16},
)
model.fit(X_tr, y_tr)
print(f"[smoke] fit OK — meta: best_epoch={model._meta['best_epoch']}")

# --- predict + eval harness ---------------------------------------------------
X_te = pack_views(Xs_te, Xq_te)
X_ch = pack_views(Xs_ch, Xq_ch)
probs = model.predict_proba(X_te)
assert probs.shape == (N_TEST,) and np.all((probs >= 0) & (probs <= 1)), "bad probabilities"
print(f"[smoke] predict_proba OK — mean p={probs.mean():.3f}")

metrics = evaluate_standard_and_evasive(model, X_te, y_te, X_ch, y_ch)
print(f"[smoke] eval harness OK — std ROC-AUC={metrics['standard_test']['roc_auc']:.3f}")

# --- save/load round-trip ------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "model.pt"
    model.save(p)
    loaded = TransformerClassifier.load(p)
    probs2 = loaded.predict_proba(X_te)
    assert np.allclose(probs, probs2, atol=1e-5), "save/load prediction mismatch"
print("[smoke] save/load OK")

print("[smoke] ALL PASSED")
