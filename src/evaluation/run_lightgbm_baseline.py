"""
Entry point: prepare static features, train LightGBM, evaluate, write paper tables.

Usage (from repo root, with venv active):
  python -m src.evaluation.run_lightgbm_baseline --config experiments/configs/lightgbm_baseline.yaml
  python -m src.evaluation.run_lightgbm_baseline --config experiments/configs/lightgbm_baseline.yaml --skip-download
  python -m src.evaluation.run_lightgbm_baseline --config experiments/configs/lightgbm_baseline.yaml --skip-vectorize
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Allow `python -m src.evaluation.run_lightgbm_baseline` from repo root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.download_ember import download_ember
from src.data.static_features import load_arrays, prepare_static_features
from src.evaluation.evasive_eval import evaluate_standard_and_evasive
from src.evaluation.write_results import write_results
from src.models.baselines.lightgbm_model import LightGBMClassifier


def _set_seed(seed: int) -> None:
    np.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate LightGBM baseline on EMBER2024.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/lightgbm_baseline.yaml",
    )
    parser.add_argument("--skip-download", action="store_true", help="Assume data/raw already has JSONL files.")
    parser.add_argument(
        "--skip-vectorize",
        action="store_true",
        help="Assume processed .npy arrays already exist.",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["experiment"]["seed"])
    _set_seed(seed)

    raw_dir = ROOT / cfg["data"]["raw_dir"]
    processed_dir = ROOT / cfg["data"]["processed_dir"]
    file_type = cfg["data"]["file_type"]

    if not args.skip_download:
        print(f"[run] downloading EMBER2024 {file_type} + challenge -> {raw_dir}")
        download_ember(raw_dir, file_type=file_type, include_challenge=True)
    else:
        print(f"[run] skip download; using {raw_dir}")

    if not args.skip_vectorize:
        print(f"[run] vectorizing static features -> {processed_dir}")
        prepare_static_features(
            raw_dir,
            processed_dir,
            filetype=file_type,
            train_max=int(cfg["data"]["train_max_samples"]),
            test_max=int(cfg["data"]["test_max_samples"]),
            train_max_files=int(cfg["data"].get("train_max_files", 8)),
            test_max_files=int(cfg["data"].get("test_max_files", 4)),
            seed=seed,
        )
    else:
        print(f"[run] skip vectorize; loading {processed_dir}")

    X_train, y_train = load_arrays(processed_dir, "train")
    X_test, y_test = load_arrays(processed_dir, "test")
    X_challenge, y_challenge = load_arrays(processed_dir, "challenge")

    print(f"[run] train={X_train.shape}, test={X_test.shape}, challenge={X_challenge.shape}")

    model = LightGBMClassifier(
        params=cfg["model"]["params"],
        val_fraction=float(cfg["model"]["val_fraction"]),
    )
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

    payload = {
        "experiment_name": cfg["experiment"]["name"],
        "description": cfg["experiment"].get("description", "").strip(),
        "config_path": str(args.config.as_posix()),
        "file_type": file_type,
        "seed": seed,
        "n_train": int(len(y_train)),
        "feature_dim": int(X_train.shape[1]),
        "train_seconds": train_seconds,
        "model_params": model.get_params(),
        "metrics": metrics,
    }

    paths = write_results(
        payload,
        ROOT / cfg["output"]["results_dir"],
        ROOT / cfg["output"]["report_table"],
    )
    print("[run] wrote paper-ready results:")
    for kind, path in paths.items():
        print(f"  - {kind}: {path}")


if __name__ == "__main__":
    main()
