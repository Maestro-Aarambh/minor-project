"""
Aligned static + byte-sequence features for the multimodal transformer.

Why it exists:
LightGBM trains on EMBER static vectors; CNN/LSTM train on histogram-derived
byte sequences. The multimodal transformer needs BOTH views of the SAME samples
in the SAME row order, with identical train/test/challenge splits so there is
no cross-split leakage and the comparison to every baseline stays fair.

Alignment guarantee:
Both vectorizers stream the same JSONL paths (same `_select_paths` file choice),
apply the same label/filetype filters in the same order, and subsample with the
same `_stratified_indices(seed)` draw. Row i of the static cache therefore
corresponds to row i of the byte cache. `load_multimodal_arrays` verifies this
by comparing labels elementwise and hard-fails on any mismatch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.byte_features import load_arrays as load_byte_arrays
from src.data.byte_features import prepare_byte_sequences
from src.data.static_features import load_arrays as load_static_arrays
from src.data.static_features import prepare_static_features

_SUBSET_FILES = ("X_{s}.npy", "y_{s}.npy")


def cache_complete(cache_dir: str | Path) -> bool:
    """True if all six .npy files (train/test/challenge) exist in cache_dir."""
    cache_dir = Path(cache_dir)
    for subset in ("train", "test", "challenge"):
        for tpl in _SUBSET_FILES:
            if not (cache_dir / tpl.format(s=subset)).exists():
                return False
    return True


def _assert_aligned(
    y_static: np.ndarray,
    y_seq: np.ndarray,
    n_static: int,
    n_seq: int,
    *,
    subset: str,
) -> None:
    if n_static != n_seq:
        raise ValueError(
            f"[multimodal] row count mismatch on '{subset}': static={n_static}, bytes={n_seq}. "
            "Rebuild both caches with the same seed, sample caps, and max_files."
        )
    if not np.array_equal(y_static, y_seq):
        n_diff = int((np.asarray(y_static) != np.asarray(y_seq)).sum())
        raise ValueError(
            f"[multimodal] label mismatch on '{subset}' ({n_diff} rows differ). "
            "Static and byte caches were built from different subsamples — rebuild both."
        )


def load_multimodal_arrays(
    static_dir: str | Path,
    bytes_dir: str | Path,
    subset: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (X_static, X_seq, y) for one split, verifying row alignment."""
    X_static, y_static = load_static_arrays(static_dir, subset)
    X_seq, y_seq = load_byte_arrays(bytes_dir, subset)
    _assert_aligned(y_static, y_seq, X_static.shape[0], X_seq.shape[0], subset=subset)
    return X_static, X_seq, y_static


def pack_views(X_static: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
    """
    Pack both views into one float32 matrix: [static | byte sequence].

    Byte values (0-255) are exactly representable in float32, so the packed
    matrix round-trips losslessly. Packing lets the multimodal model flow
    through the existing single-X evaluation harness (evasive_eval mixes test
    benign rows into the challenge set with a plain np.concatenate).
    """
    return np.hstack(
        [X_static.astype(np.float32, copy=False), X_seq.astype(np.float32, copy=False)]
    )


def unpack_views(X_packed: np.ndarray, static_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_views: -> (X_static float32, X_seq int64)."""
    X_static = np.ascontiguousarray(X_packed[:, :static_dim], dtype=np.float32)
    X_seq = np.ascontiguousarray(X_packed[:, static_dim:], dtype=np.int64)
    return X_static, X_seq


def prepare_multimodal_features(
    raw_dir: str | Path,
    static_dir: str | Path,
    bytes_dir: str | Path,
    *,
    filetype: str = "Win64",
    seq_len: int = 4096,
    train_max: int = 120_000,
    test_max: int = 40_000,
    train_max_files: int = 16,
    test_max_files: int = 8,
    seed: int = 42,
    n_workers: int = 4,
) -> None:
    """
    Build whichever cache (static / bytes) is missing, then verify alignment.

    Existing complete caches are reused untouched, so the bytes_win64 cache from
    the CNN/LSTM runs is not rebuilt.
    """
    if cache_complete(static_dir):
        print(f"[multimodal] static cache complete -> {static_dir} (reusing)", flush=True)
    else:
        print(f"[multimodal] building static cache -> {static_dir}", flush=True)
        prepare_static_features(
            raw_dir,
            static_dir,
            filetype=filetype,
            train_max=train_max,
            test_max=test_max,
            train_max_files=train_max_files,
            test_max_files=test_max_files,
            seed=seed,
            n_workers=n_workers,
        )

    if cache_complete(bytes_dir):
        print(f"[multimodal] bytes cache complete -> {bytes_dir} (reusing)", flush=True)
    else:
        print(f"[multimodal] building bytes cache -> {bytes_dir}", flush=True)
        prepare_byte_sequences(
            raw_dir,
            bytes_dir,
            filetype=filetype,
            seq_len=seq_len,
            train_max=train_max,
            test_max=test_max,
            train_max_files=train_max_files,
            test_max_files=test_max_files,
            seed=seed,
        )

    for subset in ("train", "test", "challenge"):
        X_static, X_seq, y = load_multimodal_arrays(static_dir, bytes_dir, subset)
        print(
            f"[multimodal] {subset}: aligned OK — static={X_static.shape}, "
            f"seq={X_seq.shape}, malicious={int((y == 1).sum())}, benign={int((y == 0).sum())}",
            flush=True,
        )
