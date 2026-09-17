"""
Vectorize-only entry point: raw JSONL -> bytes_win64/*.npy (no training).

Reuse the same cache for CNN, LSTM, and transformer via --skip-vectorize on those runners.

Usage (from repo root):
  python -m src.evaluation.run_vectorize_bytes --config experiments/configs/vectorize_bytes_win64.yaml --skip-download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.byte_features import load_arrays, prepare_byte_sequences
from src.data.download_ember import download_ember


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build byte-sequence .npy caches from EMBER2024 JSONL (vectorize only)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/vectorize_bytes_win64.yaml",
    )
    parser.add_argument("--skip-download", action="store_true", help="Assume data/raw already has JSONL files.")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["experiment"]["seed"])
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    processed_dir = ROOT / cfg["data"]["processed_dir"]
    file_type = cfg["data"]["file_type"]
    seq_len = int(cfg["data"]["seq_len"])

    if not args.skip_download:
        print(f"[vectorize] downloading EMBER2024 {file_type} + challenge -> {raw_dir}")
        download_ember(raw_dir, file_type=file_type, include_challenge=True)
    else:
        print(f"[vectorize] skip download; using {raw_dir}")

    print(f"[vectorize] building byte sequences -> {processed_dir}")
    prepare_byte_sequences(
        raw_dir,
        processed_dir,
        filetype=file_type,
        seq_len=seq_len,
        train_max=int(cfg["data"]["train_max_samples"]),
        test_max=int(cfg["data"]["test_max_samples"]),
        train_max_files=int(cfg["data"].get("train_max_files", 16)),
        test_max_files=int(cfg["data"].get("test_max_files", 8)),
        seed=seed,
    )

    for subset in ("train", "test", "challenge"):
        X, y = load_arrays(processed_dir, subset)
        print(
            f"[vectorize] {subset}: X={X.shape}, "
            f"malicious={int((y == 1).sum())}, benign={int((y == 0).sum())}"
        )

    print(f"[vectorize] done -> {processed_dir}")
    print("[vectorize] next: train CNN/LSTM with --skip-download --skip-vectorize")


if __name__ == "__main__":
    main()
