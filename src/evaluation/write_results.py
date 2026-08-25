"""
Write paper-ready result artifacts (JSON, CSV, Markdown).

Why it exists:
Every table in the report should be reproducible from a single results file
that records metrics, sample sizes, config path, and seed.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_ROWS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("tpr_at_fpr_1pct", "TPR @ 1% FPR"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        if isinstance(value, float) and (value != value):  # NaN
            return "—"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_results(
    payload: dict[str, Any],
    results_dir: str | Path,
    report_stem: str | Path,
) -> dict[str, Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_stem = Path(report_stem)
    report_stem.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        **payload,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    json_path = results_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    std = payload["metrics"]["standard_test"]
    eva = payload["metrics"]["evasive_challenge"]
    drop = payload["metrics"]["dropoff_standard_minus_evasive"]

    csv_path = report_stem.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Metric",
                "Standard test set",
                "Evasive challenge set",
                "Drop-off (standard − evasive)",
            ]
        )
        for key, label in METRIC_ROWS:
            writer.writerow([label, _fmt(std.get(key)), _fmt(eva.get(key)), _fmt(drop.get(key))])
        writer.writerow(
            [
                "Challenge-only detection rate",
                "—",
                _fmt(eva.get("challenge_only_detection_rate")),
                "—",
            ]
        )
        writer.writerow(["N samples (standard)", std.get("n_samples"), "", ""])
        writer.writerow(
            [
                "N samples (challenge eval)",
                "",
                eva.get("n_samples"),
                "",
            ]
        )
        writer.writerow(
            [
                "N challenge-only malicious",
                "",
                eva.get("challenge_only_n"),
                "",
            ]
        )

    md_path = report_stem.with_suffix(".md")
    lines = [
        "# LightGBM Baseline Results (EMBER2024)",
        "",
        f"**Experiment:** `{payload.get('experiment_name', '')}`  ",
        f"**Model:** LightGBM on EMBER feature v3 static vectors  ",
        f"**File type:** `{payload.get('file_type', '')}`  ",
        f"**Seed:** `{payload.get('seed', '')}`  ",
        f"**Generated (UTC):** `{payload.get('generated_at_utc', '')}`  ",
        "",
        "## Dataset sizes",
        "",
        f"- Train samples used: **{payload.get('n_train')}**",
        f"- Standard test samples used: **{std.get('n_samples')}** "
        f"(malicious={std.get('n_malicious')}, benign={std.get('n_benign')})",
        f"- Challenge-only malicious samples: **{eva.get('challenge_only_n')}**",
        f"- Challenge evaluation set size (challenge + test benign): **{eva.get('n_samples')}**",
        "",
        "## Main comparison table",
        "",
        "| Metric | Standard test | Evasive challenge | Drop-off (std − evasive) |",
        "|---|---:|---:|---:|",
    ]
    for key, label in METRIC_ROWS:
        lines.append(
            f"| {label} | {_fmt(std.get(key))} | {_fmt(eva.get(key))} | {_fmt(drop.get(key))} |"
        )
    lines.extend(
        [
            f"| Challenge-only detection rate | — | {_fmt(eva.get('challenge_only_detection_rate'))} | — |",
            "",
            "## How to read this table",
            "",
            "The research question is not 'what accuracy does LightGBM get?', but "
            "**how much does performance drop** when moving from the temporal test "
            "set to the evasive challenge set. A larger drop-off means the model "
            "relies on patterns that do not survive AV-evasive / metamorphic samples.",
            "",
            "## Notes",
            "",
            "- Challenge ROC/PR metrics follow the official EMBER2024 protocol: "
            "challenge malware is mixed with test-set benign files.",
            "- Challenge-only detection rate is measured on challenge malware alone "
            "(threshold 0.5).",
            "- Train/test sizes may be stratified subsamples for memory limits; "
            "the challenge set is used in full. Exact subsample caps are recorded "
            "in the experiment config.",
            "",
            f"Config file: `{payload.get('config_path', '')}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # Copy markdown into results_dir as well for a single artifact folder.
    results_md = results_dir / "results.md"
    results_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    results_csv = results_dir / "results.csv"
    results_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
        "results_markdown": results_md,
        "results_csv": results_csv,
    }
