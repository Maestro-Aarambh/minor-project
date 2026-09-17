# Tier 1 ablation summary

_Generated 2026-09-02T19:33:39.816501+00:00_

| Tag | Experiment | Seed | Modality | Std ROC-AUC | Chal det @ 0.5 | Evasive TPR @ 1% FPR | Train (s) | Status |
|-----|------------|-----:|----------|------------:|---------------:|---------------------:|----------:|--------|
| ablation | transformer_static_only_win64 | 42 | static_only | 0.9714 | 0.5086 | 0.3403 | 692 | done |
| ablation | transformer_bytes_only_win64 | 42 | bytes_only | 0.9583 | 0.3550 | 0.1400 | 1859 | done |
| ablation | transformer_full_win64_seed43 | 43 | full | 0.9833 | 0.5430 | 0.3022 | 1510 | done |
| ablation | transformer_full_win64_seed44 | 44 | full | 0.9723 | 0.5479 | 0.2924 | 2222 | done |
| ref:full_seed42 | transformer_win64_baseline | 42 | full | 0.9838 | 0.5835 | 0.2703 | 2681 | done |
| ref:lightgbm | lightgbm_win64_baseline | 42 | full | 0.9980 | 0.5381 | 0.4607 | 587 | done |

## How to run

```powershell
conda activate ember
cd E:\minor-project
python -m src.ablation.run_tier1 --skip-download --skip-vectorize
python -m src.evaluation.analyze_operating_points --skip-download --skip-vectorize
```
