"""
Run Tier 1 publication ablations from a manifest YAML.

Experiments: modality ablation (static_only, bytes_only) + multi-seed full transformer.

Usage (from repo root, caches already built):
  python -m src.ablation.run_tier1 --skip-download --skip-vectorize
  python -m src.ablation.run_tier1 --skip-download --skip-vectorize --only static_only
  python -m src.ablation.run_tier1 --dry-run
  python -m src.ablation.run_tier1 --summarize   # aggregate results only

Prerequisites:
  - data/processed/static_win64/
  - data/processed/bytes_win64/
  - Full seed-42 transformer already trained (reference; not re-run by manifest).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "experiments/configs/tier1_ablation_manifest.yaml"


def _load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _results_exist(cfg_path: Path) -> bool:
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    results_json = ROOT / cfg["output"]["results_dir"] / "results.json"
    model_path = ROOT / cfg["output"]["model_path"]
    return results_json.is_file() and model_path.is_file()


def _run_one(
    cfg_path: Path,
    *,
    skip_download: bool,
    skip_vectorize: bool,
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.run_transformer_baseline",
        "--config",
        str(cfg_path),
    ]
    if skip_download:
        cmd.append("--skip-download")
    if skip_vectorize:
        cmd.append("--skip-vectorize")

    print(f"\n[tier1] {'would run' if dry_run else 'running'}: {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(ROOT))


def summarize_tier1(manifest_path: Path) -> Path:
    """Aggregate ablation + reference results into one markdown table."""
    manifest = _load_manifest(manifest_path)
    rows: list[dict] = []

    def _add_from_config(cfg_rel: str, *, tag: str) -> None:
        cfg_path = ROOT / cfg_rel
        if not cfg_path.is_file():
            return
        with cfg_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        results_json = ROOT / cfg["output"]["results_dir"] / "results.json"
        if not results_json.is_file():
            rows.append(
                {
                    "tag": tag,
                    "name": cfg["experiment"]["name"],
                    "seed": cfg["experiment"]["seed"],
                    "modality": cfg.get("model", {}).get("modality", "full"),
                    "status": "pending",
                }
            )
            return
        payload = json.loads(results_json.read_text(encoding="utf-8"))
        std = payload["metrics"]["standard_test"]
        eva = payload["metrics"]["evasive_challenge"]
        rows.append(
            {
                "tag": tag,
                "name": payload.get("experiment_name", cfg["experiment"]["name"]),
                "seed": payload.get("seed", cfg["experiment"]["seed"]),
                "modality": payload.get("model_params", {}).get("modality", "full"),
                "status": "done",
                "std_roc_auc": std.get("roc_auc"),
                "chal_det_05": eva.get("challenge_only_detection_rate"),
                "eva_tpr_1pct": eva.get("tpr_at_fpr_1pct"),
                "train_s": payload.get("train_seconds"),
            }
        )

    for cfg_rel in manifest.get("experiments", []):
        _add_from_config(cfg_rel, tag="ablation")

    for key, cfg_rel in manifest.get("reference", {}).items():
        _add_from_config(cfg_rel, tag=f"ref:{key}")

    out_dir = ROOT / "report/tables/ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tier1_ablation_summary.md"

    lines = [
        "# Tier 1 ablation summary",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        "",
        "| Tag | Experiment | Seed | Modality | Std ROC-AUC | Chal det @ 0.5 | Evasive TPR @ 1% FPR | Train (s) | Status |",
        "|-----|------------|-----:|----------|------------:|---------------:|---------------------:|----------:|--------|",
    ]

    def _fmt(v: object) -> str:
        if v is None:
            return "—"
        try:
            f = float(v)
            if f != f:
                return "—"
            return f"{f:.4f}" if abs(f) <= 1.5 else f"{f:.0f}"
        except (TypeError, ValueError):
            return str(v)

    for r in rows:
        lines.append(
            "| {tag} | {name} | {seed} | {modality} | {std_roc} | {chal} | {tpr} | {train} | {status} |".format(
                tag=r["tag"],
                name=r["name"],
                seed=r["seed"],
                modality=r.get("modality", "—"),
                std_roc=_fmt(r.get("std_roc_auc")),
                chal=_fmt(r.get("chal_det_05")),
                tpr=_fmt(r.get("eva_tpr_1pct")),
                train=_fmt(r.get("train_s")),
                status=r["status"],
            )
        )

    lines.extend(
        [
            "",
            "## How to run",
            "",
            "```powershell",
            "conda activate ember",
            "cd E:\\minor-project",
            "python -m src.ablation.run_tier1 --skip-download --skip-vectorize",
            "python -m src.evaluation.analyze_operating_points --skip-download --skip-vectorize",
            "```",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[tier1] summary -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tier 1 ablation manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="YAML manifest listing ablation configs.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-vectorize", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Substring filter on config path or experiment name (e.g. static_only, seed43).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs whose results.json and model checkpoint already exist.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Only aggregate results into report/tables/ablations/tier1_ablation_summary.md.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    if args.summarize:
        summarize_tier1(manifest_path)
        return

    manifest = _load_manifest(manifest_path)
    configs = [ROOT / p for p in manifest.get("experiments", [])]
    filt = args.only.strip().lower()
    if filt:
        configs = [
            p
            for p in configs
            if filt in p.as_posix().lower()
            or filt in yaml.safe_load(p.read_text(encoding="utf-8"))["experiment"]["name"].lower()
        ]

    if not configs:
        raise SystemExit("No experiments matched the manifest / --only filter.")

    print(f"[tier1] manifest: {manifest_path}")
    print(f"[tier1] {len(configs)} experiment(s) queued")

    failures = 0
    for cfg_path in configs:
        if not cfg_path.is_file():
            print(f"[tier1] SKIP missing config: {cfg_path}")
            failures += 1
            continue
        if args.skip_existing and _results_exist(cfg_path):
            print(f"[tier1] SKIP existing results: {cfg_path.name}")
            continue
        rc = _run_one(
            cfg_path,
            skip_download=args.skip_download,
            skip_vectorize=args.skip_vectorize,
            dry_run=args.dry_run,
        )
        if rc != 0:
            failures += 1
            print(f"[tier1] FAILED (exit {rc}): {cfg_path.name}")

    if not args.dry_run:
        summarize_tier1(manifest_path)

    if failures:
        raise SystemExit(f"{failures} experiment(s) failed.")
    print("[tier1] all queued experiments finished successfully.")


if __name__ == "__main__":
    main()
