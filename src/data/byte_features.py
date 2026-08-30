"""
Byte-sequence features for CNN / LSTM / transformer baselines.

Why it exists:
EMBER2024 distributes JSONL feature records, not raw PE binaries. We build a
fixed-length byte sequence from each file's 256-bin byte histogram: bytes are
repeated in proportion to their frequency (deterministic, no randomness). This
approximates the byte composition as a 1D signal for sequence models.

Limitation (document in the paper): this is NOT the original file byte order —
it is a histogram-derived proxy. Raw-byte MalConv-style input would require
VirusTotal API access to download binaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from thrember.model import gather_feature_paths, raw_feature_iterator

from src.data.static_features import (
    _normalize_filetype,
    _select_paths,
    _stratified_indices,
    load_arrays,
    save_arrays,
)


def histogram_to_sequence(histogram: list[int] | np.ndarray, seq_len: int) -> np.ndarray:
    """
    Expand a 256-bin byte histogram into a length-seq_len sequence of byte values 0–255.

    Each byte value i appears roughly (count_i / total) * seq_len times, sorted by
    byte value so local CNN/LSTM windows see frequency structure.
    """
    counts = np.asarray(histogram, dtype=np.float64)
    if counts.shape[0] != 256:
        raise ValueError(f"Expected 256-bin histogram, got {counts.shape[0]}")
    total = counts.sum()
    if total <= 0:
        return np.zeros(seq_len, dtype=np.int64)

    alloc = np.floor(counts / total * seq_len).astype(np.int64)
    diff = int(seq_len - alloc.sum())
    if diff > 0:
        order = np.argsort(-counts)
        for idx in order:
            if diff == 0:
                break
            alloc[idx] += 1
            diff -= 1
    elif diff < 0:
        order = np.argsort(-alloc)
        for idx in order:
            if diff == 0:
                break
            if alloc[idx] > 0:
                alloc[idx] -= 1
                diff += 1

    parts = [
        np.full(int(c), i, dtype=np.int64)
        for i, c in enumerate(alloc)
        if c > 0
    ]
    seq = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    if len(seq) < seq_len:
        seq = np.pad(seq, (0, seq_len - len(seq)))
    elif len(seq) > seq_len:
        seq = seq[:seq_len]
    return seq


def _line_to_sequence(line: str, seq_len: int) -> tuple[np.ndarray, int]:
    raw = json.loads(line)
    hist = raw.get("histogram")
    if hist is None:
        raise ValueError("JSONL record missing histogram field")
    seq = histogram_to_sequence(hist, seq_len)
    label = raw.get("label")
    if label is None:
        return seq, -1
    return seq, int(label)


def subset_to_sequences(
    data_dir: str | Path,
    subset: str,
    *,
    seq_len: int = 4096,
    filetype: str | None = None,
    max_samples: int | None = None,
    max_files: int | None = None,
    seed: int = 42,
    filter_json_filetype: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one subset as (N, seq_len) int64 sequences and labels."""
    data_dir = Path(data_dir)
    paths = _select_paths(data_dir, subset, filetype, max_files)
    wanted = _normalize_filetype(filetype) if filter_json_filetype else None

    labels: list[int] = []
    lines: list[str] = []
    for line in raw_feature_iterator(paths):
        raw = json.loads(line)
        if wanted is not None:
            row_ft = _normalize_filetype(str(raw.get("file_type", "")))
            if row_ft != wanted:
                continue
        label = raw.get("label")
        if label is None:
            continue
        labels.append(int(label))
        lines.append(line)

    y_all = np.asarray(labels, dtype=np.int32)
    keep = _stratified_indices(y_all, max_samples, seed)
    selected = [lines[i] for i in keep]

    n = len(selected)
    X = np.zeros((n, seq_len), dtype=np.int64)
    y = np.zeros(n, dtype=np.int32)
    for i, line in enumerate(selected):
        seq, label = _line_to_sequence(line, seq_len)
        X[i] = seq
        y[i] = label

    mask = y != -1
    return X[mask], y[mask]


def prepare_byte_sequences(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    filetype: str = "Win64",
    seq_len: int = 4096,
    train_max: int = 120_000,
    test_max: int = 40_000,
    train_max_files: int = 16,
    test_max_files: int = 8,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build train / test / challenge byte sequences and cache as .npy."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (filetype, train_max, train_max_files, False),
        "test": (filetype, test_max, test_max_files, False),
        "challenge": (filetype, None, None, True),
    }
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for subset, (ft, max_n, max_files, filter_json) in splits.items():
        print(f"[byte_features] preparing {subset} (seq_len={seq_len}, max={max_n})...", flush=True)
        X, y = subset_to_sequences(
            raw_dir,
            subset,
            seq_len=seq_len,
            filetype=ft,
            max_samples=max_n,
            max_files=max_files,
            seed=seed,
            filter_json_filetype=filter_json,
        )
        save_arrays(X, y, processed_dir, subset)
        print(
            f"[byte_features] {subset}: X={X.shape}, "
            f"malicious={int((y == 1).sum())}, benign={int((y == 0).sum())}",
            flush=True,
        )
        out[subset] = (X, y)
    return out


# Re-export for sequence runners
__all__ = [
    "histogram_to_sequence",
    "prepare_byte_sequences",
    "subset_to_sequences",
    "load_arrays",
]
