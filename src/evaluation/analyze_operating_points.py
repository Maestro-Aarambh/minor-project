"""
Val-calibrated thresholds and LightGBM + transformer ensemble analysis.

All tuning uses the validation split carved from TRAIN ONLY (same seed/stratify
as model training). Test and challenge are evaluated once with fixed thresholds.

Usage:
  python -m src.evaluation.analyze_operating_points --skip-download --skip-vectorize
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.multimodal_features import load_multimodal_arrays, pack_views
from src.evaluation.metrics import binary_metrics
from src.models.baselines.lightgbm_model import LightGBMClassifier
from src.models.transformer.classifier import TransformerClassifier

DEFAULT_CONFIG = ROOT / "experiments/configs/analyze_operating_points.yaml"


def threshold_at_benign_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Score threshold where ~target_fpr of benign validation samples are flagged."""
    benign_scores = np.asarray(y_score, dtype=np.float64)[np.asarray(y_true) == 0]
    if len(benign_scores) == 0:
        return 0.5
    # Higher score => malicious; FPR = P(score >= t | benign).
    q = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
    return float(np.quantile(benign_scores, q))


def best_ensemble_weight(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float]:
    """Grid-search blend weight w on validation ROC-AUC: w*A + (1-w)*B."""
    best_w, best_auc = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * scores_a + (1.0 - w) * scores_b
        auc = float(roc_auc_score(y_val, blend))
        if auc > best_auc:
            best_w, best_auc = float(w), auc
    return best_w, best_auc


def _eval_split(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    challenge_only: bool = False,
) -> dict:
    m = binary_metrics(y_true, y_score, threshold=threshold)
    if challenge_only and int(np.sum(y_true == 0)) == 0:
        m["challenge_only_detection_rate"] = float(np.mean(y_score >= threshold))
        m["challenge_only_n"] = int(len(y_true))
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold + ensemble operating-point analysis.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-download", action="store_true", help="Unused; kept for CLI parity.")
    parser.add_argument("--skip-vectorize", action="store_true", help="Unused; kept for CLI parity.")
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["experiment"]["seed"])
    val_fraction = float(cfg["data"]["val_fraction"])
    target_fpr = float(cfg["evaluation"]["target_fpr"])
    threshold_default = float(cfg["evaluation"]["threshold_default"])
    mix_benign = bool(cfg["evaluation"]["mix_test_benign_into_challenge"])

    static_dir = ROOT / cfg["data"]["static_dir"]
    bytes_dir = ROOT / cfg["data"]["bytes_dir"]

    Xs_train, Xq_train, y_train = load_multimodal_arrays(static_dir, bytes_dir, "train")
    Xs_test, Xq_test, y_test = load_multimodal_arrays(static_dir, bytes_dir, "test")
    Xs_chal, Xq_chal, y_chal = load_multimodal_arrays(static_dir, bytes_dir, "challenge")

    static_dim = int(Xs_train.shape[1])
    idx_tr, idx_val = train_test_split(
        np.arange(len(y_train)),
        test_size=val_fraction,
        stratify=y_train,
        random_state=seed,
    )

    lgbm = LightGBMClassifier.load(ROOT / cfg["models"]["lightgbm"])
    transformer = TransformerClassifier.load(ROOT / cfg["models"]["transformer"])

    # Validation scores (tuning only).
    lgbm_val = lgbm.predict_proba(Xs_train[idx_val])
    tr_val = transformer.predict_proba(pack_views(Xs_train[idx_val], Xq_train[idx_val]))
    y_val = y_train[idx_val]

    lgbm_t_cal = threshold_at_benign_fpr(y_val, lgbm_val, target_fpr)
    tr_t_cal = threshold_at_benign_fpr(y_val, tr_val, target_fpr)
    ens_w, ens_val_auc = best_ensemble_weight(lgbm_val, tr_val, y_val)
    ens_val = ens_w * lgbm_val + (1.0 - ens_w) * tr_val
    ens_t_cal = threshold_at_benign_fpr(y_val, ens_val, target_fpr)

    # Test scores.
    lgbm_test = lgbm.predict_proba(Xs_test)
    tr_test = transformer.predict_proba(pack_views(Xs_test, Xq_test))
    ens_test = ens_w * lgbm_test + (1.0 - ens_w) * tr_test

    # Challenge eval (mixed benign for ROC metrics).
    if mix_benign:
        Xs_chal_eval = np.concatenate([Xs_test[y_test == 0], Xs_chal], axis=0)
        Xq_chal_eval = np.concatenate([Xq_test[y_test == 0], Xq_chal], axis=0)
        y_chal_eval = np.concatenate([y_test[y_test == 0], y_chal], axis=0)
    else:
        Xs_chal_eval, Xq_chal_eval, y_chal_eval = Xs_chal, Xq_chal, y_chal

    lgbm_chal = lgbm.predict_proba(Xs_chal_eval)
    tr_chal = transformer.predict_proba(pack_views(Xs_chal_eval, Xq_chal_eval))
    ens_chal = ens_w * lgbm_chal + (1.0 - ens_w) * tr_chal
    lgbm_chal_only = lgbm.predict_proba(Xs_chal)
    tr_chal_only = transformer.predict_proba(pack_views(Xs_chal, Xq_chal))
    ens_chal_only = ens_w * lgbm_chal_only + (1.0 - ens_w) * tr_chal_only

    scenarios = {
        "lightgbm": (lgbm_test, lgbm_chal, lgbm_chal_only),
        "transformer": (tr_test, tr_chal, tr_chal_only),
        "ensemble": (ens_test, ens_chal, ens_chal_only),
    }
    thresholds = {
        "default_0.5": threshold_default,
        "val_calibrated_1pct_fpr": {
            "lightgbm": lgbm_t_cal,
            "transformer": tr_t_cal,
            "ensemble": ens_t_cal,
        },
    }

    results: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path.as_posix()),
        "seed": seed,
        "val_fraction": val_fraction,
        "target_fpr": target_fpr,
        "ensemble_weight_lightgbm": ens_w,
        "ensemble_val_roc_auc": ens_val_auc,
        "thresholds": thresholds,
        "splits": {},
    }

    for model_name, (test_scores, chal_scores, chal_only_scores) in scenarios.items():
        t_def = threshold_default
        t_cal = thresholds["val_calibrated_1pct_fpr"][model_name]
        results["splits"][model_name] = {
            "standard_test@0.5": _eval_split(y_test, test_scores, threshold=t_def),
            "standard_test@calibrated": _eval_split(y_test, test_scores, threshold=t_cal),
            "evasive_challenge@0.5": _eval_split(y_chal_eval, chal_scores, threshold=t_def),
            "evasive_challenge@calibrated": _eval_split(y_chal_eval, chal_scores, threshold=t_cal),
            "challenge_only@0.5": _eval_split(
                y_chal, chal_only_scores, threshold=t_def, challenge_only=True
            ),
            "challenge_only@calibrated": _eval_split(
                y_chal, chal_only_scores, threshold=t_cal, challenge_only=True
            ),
        }

    out_dir = ROOT / cfg["output"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "operating_points.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_path = ROOT / cfg["output"]["report_table"]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Tier 1 operating-point analysis",
        "",
        f"_Generated {results['generated_at_utc']}_",
        "",
        "Validation-only tuning (10% stratified holdout from train, seed "
        f"{seed}). Target benign FPR: **{target_fpr:.0%}**.",
        "",
        f"**Ensemble weight (LightGBM):** {ens_w:.2f} (val ROC-AUC {ens_val_auc:.4f})",
        "",
        "## Calibrated thresholds (validation)",
        "",
        "| Model | Threshold @ 1% val FPR |",
        "|-------|-----------------------:|",
        f"| LightGBM | {lgbm_t_cal:.4f} |",
        f"| Transformer | {tr_t_cal:.4f} |",
        f"| Ensemble | {ens_t_cal:.4f} |",
        "",
        "## Challenge-only detection rate",
        "",
        "| Model | @ 0.5 | @ val-calibrated |",
        "|-------|------:|-----------------:|",
    ]
    for name in ("lightgbm", "transformer", "ensemble"):
        a = results["splits"][name]["challenge_only@0.5"].get("challenge_only_detection_rate", float("nan"))
        b = results["splits"][name]["challenge_only@calibrated"].get(
            "challenge_only_detection_rate", float("nan")
        )
        md_lines.append(f"| {name} | {a:.4f} | {b:.4f} |")

    md_lines.extend(
        [
            "",
            "## Evasive challenge TPR @ 1% FPR (mixed with test benign)",
            "",
            "| Model | @ 0.5 threshold | @ val-calibrated threshold |",
            "|-------|----------------:|---------------------------:|",
        ]
    )
    for name in ("lightgbm", "transformer", "ensemble"):
        a = results["splits"][name]["evasive_challenge@0.5"].get("tpr_at_fpr_1pct", float("nan"))
        b = results["splits"][name]["evasive_challenge@calibrated"].get("tpr_at_fpr_1pct", float("nan"))
        md_lines.append(f"| {name} | {a:.4f} | {b:.4f} |")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[op-points] wrote {json_path}")
    print(f"[op-points] wrote {md_path}")


if __name__ == "__main__":
    main()
