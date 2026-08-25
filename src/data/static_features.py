"""
EMBER static (thrember / EMBER feature v3) vectorization and loading.

Why it exists:
Tree baselines (LightGBM) need the engineered PE feature vectors, not raw
bytes. We vectorize from EMBER2024 JSONL using thrember's PEFeatureExtractor,
with file-count and sample caps so a laptop run finishes in reasonable time.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from thrember.features import PEFeatureExtractor
from thrember.model import gather_feature_paths, raw_feature_iterator


def _feature_dim() -> int:
    return PEFeatureExtractor().dim


def _normalize_filetype(value: str | None) -> str | None:
    if value is None:
        return None
    mapping = {
        "dot_net": "dotnet",
        ".net": "dotnet",
        "dotnet": "dotnet",
        "win32": "win32",
        "win64": "win64",
    }
    key = value.strip().lower().replace(" ", "_")
    return mapping.get(key, key)


def _stratified_indices(y: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    if max_samples is None or max_samples >= n:
        return np.arange(n)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0:
        return rng.choice(idx1, size=min(max_samples, len(idx1)), replace=False)
    if len(idx1) == 0:
        return rng.choice(idx0, size=min(max_samples, len(idx0)), replace=False)

    n1 = int(round(max_samples * (len(idx1) / n)))
    n1 = max(1, min(len(idx1), n1))
    n0 = max_samples - n1
    n0 = max(1, min(len(idx0), n0))
    while n0 + n1 > max_samples:
        if n0 >= n1 and n0 > 1:
            n0 -= 1
        elif n1 > 1:
            n1 -= 1
        else:
            break

    chosen0 = rng.choice(idx0, size=n0, replace=False)
    chosen1 = rng.choice(idx1, size=n1, replace=False)
    out = np.concatenate([chosen0, chosen1])
    rng.shuffle(out)
    return out


def _vectorize_line(line: str) -> tuple[np.ndarray, int]:
    extractor = PEFeatureExtractor()
    raw = json.loads(line)
    vec = np.asarray(extractor.process_raw_features(raw), dtype=np.float32)
    label = -1 if raw.get("label") is None else int(raw["label"])
    return vec, label


def _select_paths(
    data_dir: Path,
    subset: str,
    filetype: str | None,
    max_files: int | None,
) -> list[Path]:
    path_filetype = None if subset == "challenge" else filetype
    paths = gather_feature_paths(data_dir, subset, filetype=path_filetype)
    if max_files is not None and max_files < len(paths):
        # Evenly spaced weeks across the available temporal range.
        idx = np.linspace(0, len(paths) - 1, max_files, dtype=int)
        paths = [paths[i] for i in idx]
    return paths


def vectorize_subset_to_arrays(
    data_dir: str | Path,
    subset: str,
    *,
    filetype: str | None = None,
    max_samples: int | None = None,
    max_files: int | None = None,
    seed: int = 42,
    n_workers: int = 4,
    filter_json_filetype: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stream selected JSONL files once, keep eligible lines, stratified-subsample,
    then vectorize in parallel.
    """
    data_dir = Path(data_dir)
    paths = _select_paths(data_dir, subset, filetype, max_files)
    wanted = _normalize_filetype(filetype) if filter_json_filetype else None
    print(f"[static_features] {subset}: reading {len(paths)} jsonl files...", flush=True)

    labels: list[int] = []
    lines: list[str] = []
    for line in raw_feature_iterator(paths):
        obj = json.loads(line)
        if wanted is not None:
            row_ft = _normalize_filetype(str(obj.get("file_type", "")))
            if row_ft != wanted:
                continue
        label = obj.get("label")
        if label is None:
            continue
        labels.append(int(label))
        lines.append(line)

    print(f"[static_features] {subset}: loaded {len(lines)} eligible rows", flush=True)
    y_all = np.asarray(labels, dtype=np.int32)
    keep = _stratified_indices(y_all, max_samples, seed)
    selected_lines = [lines[i] for i in keep]
    # Free raw buffers before vectorization.
    del lines, labels

    dim = _feature_dim()
    n = len(selected_lines)
    X = np.zeros((n, dim), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)

    print(f"[static_features] {subset}: vectorizing {n} rows with {n_workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_vectorize_line, selected_lines, chunksize=16))

    for i, (vec, label) in enumerate(results):
        X[i] = vec
        y[i] = label

    mask = y != -1
    return X[mask], y[mask]


def save_arrays(X: np.ndarray, y: np.ndarray, out_dir: str | Path, subset: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"X_{subset}.npy", X)
    np.save(out_dir / f"y_{subset}.npy", y)


def load_arrays(data_dir: str | Path, subset: str) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    return np.load(data_dir / f"X_{subset}.npy"), np.load(data_dir / f"y_{subset}.npy")


def prepare_static_features(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    filetype: str = "Dot_Net",
    train_max: int = 40_000,
    test_max: int = 15_000,
    train_max_files: int = 8,
    test_max_files: int = 4,
    seed: int = 42,
    n_workers: int = 4,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build train / test / challenge static-feature arrays and cache them as .npy.

    Challenge rows are filtered to the same file_type as training.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        # subset: (filetype, max_samples, max_files, filter_json_filetype)
        "train": (filetype, train_max, train_max_files, False),
        "test": (filetype, test_max, test_max_files, False),
        "challenge": (filetype, None, None, True),
    }
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for subset, (ft, max_n, max_files, filter_json) in splits.items():
        print(f"[static_features] preparing {subset}...", flush=True)
        X, y = vectorize_subset_to_arrays(
            raw_dir,
            subset,
            filetype=ft,
            max_samples=max_n,
            max_files=max_files,
            seed=seed,
            n_workers=n_workers,
            filter_json_filetype=filter_json,
        )
        save_arrays(X, y, processed_dir, subset)
        print(
            f"[static_features] {subset}: X={X.shape}, "
            f"malicious={int((y == 1).sum())}, benign={int((y == 0).sum())}",
            flush=True,
        )
        out[subset] = (X, y)
    return out
