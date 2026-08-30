"""
Entry point: byte sequences -> CNN or LSTM -> evaluate -> paper tables.

Usage:
  python -m src.evaluation.run_sequence_baseline --config experiments/configs/cnn_win64_baseline.yaml
  python -m src.evaluation.run_sequence_baseline --config experiments/configs/lstm_win64_baseline.yaml
  python -m src.evaluation.run_sequence_baseline --config ... --skip-download --skip-vectorize
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.byte_features import load_arrays, prepare_byte_sequences
from src.data.download_ember import download_ember
from src.evaluation.evasive_eval import evaluate_standard_and_evasive
from src.evaluation.write_results import write_results
from src.models.baselines.cnn_model import CNNClassifier
from src.models.baselines.lstm_model import LSTMClassifier
from src.models.baselines import training


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    training.set_seed(seed)


def _build_model(cfg: dict, seq_len: int, seed: int):
    mcfg = cfg["model"]
    train_cfg = {**mcfg.get("train", {}), "seed": seed}
    model_type = mcfg["type"].lower()
    if model_type == "cnn":
        return CNNClassifier(
            seq_len=seq_len,
            embed_dim=int(mcfg.get("embed_dim", 64)),
            num_filters=tuple(mcfg.get("num_filters", [128, 256, 256])),
            kernel_size=int(mcfg.get("kernel_size", 7)),
            dropout=float(mcfg.get("dropout", 0.2)),
            train_config=train_cfg,
        )
    if model_type == "lstm":
        return LSTMClassifier(
            seq_len=seq_len,
            embed_dim=int(mcfg.get("embed_dim", 64)),
            hidden_dim=int(mcfg.get("hidden_dim", 128)),
            num_layers=int(mcfg.get("num_layers", 2)),
            dropout=float(mcfg.get("dropout", 0.2)),
            bidirectional=bool(mcfg.get("bidirectional", True)),
            train_config=train_cfg,
        )
    raise ValueError(f"Unknown model.type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate CNN or LSTM on EMBER2024 byte sequences.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-vectorize", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["experiment"]["seed"])
    _set_seed(seed)

    raw_dir = ROOT / cfg["data"]["raw_dir"]
    processed_dir = ROOT / cfg["data"]["processed_dir"]
    file_type = cfg["data"]["file_type"]
    seq_len = int(cfg["data"]["seq_len"])

    if not args.skip_download:
        print(f"[run] downloading EMBER2024 {file_type} + challenge -> {raw_dir}")
        download_ember(raw_dir, file_type=file_type, include_challenge=True)
    else:
        print(f"[run] skip download; using {raw_dir}")

    if not args.skip_vectorize:
        print(f"[run] building byte sequences -> {processed_dir}")
        prepare_byte_sequences(
            raw_dir,
            processed_dir,
            filetype=file_type,
            seq_len=seq_len,
            train_max=int(cfg["data"]["train_max_samples"]),
            test_max=int(cfg["data"]["test_max_samples"]),
            train_max_files=int(cfg["data"].get("train_max_files", 16)),
            test_max_files=int(cfg["data"].get("test_max_files", 4)),
            seed=seed,
        )
    else:
        print(f"[run] skip vectorize; loading {processed_dir}")

    X_train, y_train = load_arrays(processed_dir, "train")
    X_test, y_test = load_arrays(processed_dir, "test")
    X_challenge, y_challenge = load_arrays(processed_dir, "challenge")

    print(f"[run] train={X_train.shape}, test={X_test.shape}, challenge={X_challenge.shape}")

    model = _build_model(cfg, seq_len, seed)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0
    print(f"[run] training finished in {train_seconds:.1f}s")

    model_path = ROOT / cfg["output"]["model_path"]
    model.save(model_path)
    print(f"[run] saved model -> {model_path}")

    metrics = evaluate_standard_and_evasive(
        model,
        X_test,
        y_test,
        X_challenge,
        y_challenge,
        threshold=float(cfg["evaluation"]["threshold"]),
        mix_test_benign_into_challenge=bool(cfg["evaluation"]["mix_test_benign_into_challenge"]),
    )

    out = cfg["output"]
    payload = {
        "experiment_name": cfg["experiment"]["name"],
        "description": cfg["experiment"].get("description", "").strip(),
        "config_path": str(args.config.as_posix()),
        "file_type": file_type,
        "seed": seed,
        "n_train": int(len(y_train)),
        "seq_len": seq_len,
        "feature_dim": seq_len,
        "train_seconds": train_seconds,
        "model_params": model.get_params(),
        "model_label": out.get("model_label", cfg["model"]["type"]),
        "input_description": out.get("input_description", "histogram-derived byte sequence"),
        "report_title": out.get("report_title", "Baseline Results (EMBER2024)"),
        "metrics": metrics,
    }

    paths = write_results(
        payload,
        ROOT / out["results_dir"],
        ROOT / out["report_table"],
    )
    print("[run] wrote paper-ready results:")
    for kind, path in paths.items():
        print(f"  - {kind}: {path}")


if __name__ == "__main__":
    main()
