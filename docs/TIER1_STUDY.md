# Tier 1 ablation study (publication)

Tier 1 extends the Win64 baseline matrix with **formal modality ablations**, **multi-seed replication**, and **val-calibrated operating points**.

## Prerequisites

Both feature caches must exist (already built locally):

- `data/processed/static_win64/`
- `data/processed/bytes_win64/`

Reference baselines (already completed, not in the run manifest):

- Full transformer seed 42 → `experiments/results/transformer_win64_baseline/`
- LightGBM → `experiments/results/lightgbm_win64_baseline/`

## 1. Modality ablation + multi-seed

| Config | Modality | Seed | Est. GPU time |
|--------|----------|-----:|--------------:|
| `ablations/transformer_static_only_win64.yaml` | static_only | 42 | ~15 min |
| `ablations/transformer_bytes_only_win64.yaml` | bytes_only | 42 | ~45 min |
| `ablations/transformer_full_win64_seed43.yaml` | full | 43 | ~45 min |
| `ablations/transformer_full_win64_seed44.yaml` | full | 44 | ~45 min |

### Run all (recommended)

```powershell
conda activate ember
cd E:\minor-project

python -m src.ablation.run_tier1 --skip-download --skip-vectorize
```

### Run one experiment

```powershell
python -m src.evaluation.run_transformer_baseline `
  --config experiments/configs/ablations/transformer_static_only_win64.yaml `
  --skip-download --skip-vectorize
```

### Filter / dry-run

```powershell
python -m src.ablation.run_tier1 --dry-run
python -m src.ablation.run_tier1 --skip-download --skip-vectorize --only static_only
python -m src.ablation.run_tier1 --skip-download --skip-vectorize --skip-existing
python -m src.ablation.run_tier1 --summarize
```

Results land under `experiments/results/ablations/` and `report/tables/ablations/`.  
Summary table: `report/tables/ablations/tier1_ablation_summary.md`.

## 2. Threshold + ensemble analysis

Uses **validation split from train only** (no test/challenge leakage):

- Per-model threshold tuned for ~1% benign FPR on validation
- LightGBM + transformer ensemble weight grid-searched on validation ROC-AUC

```powershell
python -m src.evaluation.analyze_operating_points --skip-download --skip-vectorize
```

Outputs:

- `experiments/results/tier1_operating_points/operating_points.json`
- `report/tables/ablations/tier1_operating_points.md`

## 3. Update the report

After runs finish:

1. Open `report/tables/ablations/tier1_ablation_summary.md` — paste into **Section 8** of `report/REPORT_DRAFT.md`.
2. Paste operating-point table from `tier1_operating_points.md`.
3. Replace limitation #4 (single seed) and #5 (fixed 0.5 threshold) if multi-seed / calibration change the story.

## Manifest

All Tier 1 experiment paths are listed in `experiments/configs/tier1_ablation_manifest.yaml`.
