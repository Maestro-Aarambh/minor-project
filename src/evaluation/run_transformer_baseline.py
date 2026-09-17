"""
Entry point: static + byte views -> multimodal transformer -> evaluate -> paper tables.

Usage (from repo root):
  python -m src.evaluation.run_transformer_baseline --config experiments/configs/transformer_win64_baseline.yaml --skip-download
  python -m src.evaluation.run_transformer_baseline --config ... --skip-download --skip-vectorize

Notes:
- --skip-vectorize assumes BOTH caches exist (static_win64 and bytes_win64).
- Without it, only the missing cache is built (existing caches are reused).
- Alignment between the two caches is always verified before training.
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

from src.data.download_ember import download_ember
from src.data.multimodal_features import (
    load_multimodal_arrays,
    pack_views,
    prepare_multimodal_features,
)
from src.evaluation.evasive_eval import evaluate_standard_and_evasive
from src.evaluation.write_results import write_results
from src.models.baselines import training
from src.models.transformer.classifier import TransformerClassifier


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    training.set_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the multimodal transformer on EMBER2024."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/transformer_win64_baseline.yaml",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--skip-vectorize",
        action="store_true",
        help="Assume static and byte caches already exist.",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["experiment"]["seed"])
    _set_seed(seed)

    raw_dir = ROOT / cfg["data"]["raw_dir"]
    static_dir = ROOT / cfg["data"]["static_dir"]
    bytes_dir = ROOT / cfg["data"]["bytes_dir"]
    file_type = cfg["data"]["file_type"]
    seq_len = int(cfg["data"]["seq_len"])

    if not args.skip_download:
        print(f"[run] downloading EMBER2024 {file_type} + challenge -> {raw_dir}")
        download_ember(raw_dir, file_type=file_type, include_challenge=True)
    else:
        print(f"[run] skip download; using {raw_dir}")

    if not args.skip_vectorize:
        prepare_multimodal_features(
            raw_dir,
            static_dir,
            bytes_dir,
            filetype=file_type,
            seq_len=seq_len,
            train_max=int(cfg["data"]["train_max_samples"]),
            test_max=int(cfg["data"]["test_max_samples"]),
            train_max_files=int(cfg["data"].get("train_max_files", 16)),
            test_max_files=int(cfg["data"].get("test_max_files", 8)),
            seed=seed,
        )
    else:
        print(f"[run] skip vectorize; using {static_dir} + {bytes_dir}")

    Xs_train, Xq_train, y_train = load_multimodal_arrays(static_dir, bytes_dir, "train")
    Xs_test, Xq_test, y_test = load_multimodal_arrays(static_dir, bytes_dir, "test")
    Xs_chal, Xq_chal, y_chal = load_multimodal_arrays(static_dir, bytes_dir, "challenge")

    static_dim = int(Xs_train.shape[1])
    print(
        f"[run] train={Xq_train.shape} (+static {static_dim}), "
        f"test={Xq_test.shape}, challenge={Xq_chal.shape}"
    )

    X_train = pack_views(Xs_train, Xq_train)
    X_test = pack_views(Xs_test, Xq_test)
    X_challenge = pack_views(Xs_chal, Xq_chal)
    del Xs_train, Xq_train, Xs_test, Xq_test, Xs_chal, Xq_chal

    mcfg = cfg["model"]
    train_cfg = {**mcfg.get("train", {}), "seed": seed}
    model = TransformerClassifier(
        static_dim=static_dim,
        seq_len=seq_len,
        patch_size=int(mcfg.get("patch_size", 16)),
        byte_embed_dim=int(mcfg.get("byte_embed_dim", 32)),
        d_model=int(mcfg.get("d_model", 128)),
        nhead=int(mcfg.get("nhead", 4)),
        num_layers=int(mcfg.get("num_layers", 3)),
        ff_dim=int(mcfg.get("ff_dim", 256)),
        dropout=float(mcfg.get("dropout", 0.2)),
        modality=str(mcfg.get("modality", "full")),
        train_config=train_cfg,
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
        y_chal,
        threshold=float(cfg["evaluation"]["threshold"]),
        mix_test_benign_into_challenge=bool(
            cfg["evaluation"]["mix_test_benign_into_challenge"]
        ),
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
        "feature_dim": int(static_dim + seq_len),
        "train_seconds": train_seconds,
        "model_params": model.get_params(),
        "model_label": out.get("model_label", "multimodal transformer"),
        "input_description": out.get("input_description", "static + byte sequence"),
        "report_title": out.get("report_title", "Multimodal Transformer Results (EMBER2024)"),
        "metrics": metrics,
    }

    paths = write_results(payload, ROOT / out["results_dir"], ROOT / out["report_table"])
    print("[run] wrote paper-ready results:")
    for kind, path in paths.items():
        print(f"  - {kind}: {path}")


if __name__ == "__main__":
    main()
