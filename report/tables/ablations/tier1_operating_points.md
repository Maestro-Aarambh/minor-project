# Tier 1 operating-point analysis

_Generated 2026-09-02T06:55:11.954881+00:00_

Validation-only tuning (10% stratified holdout from train, seed 42). Target benign FPR: **1%**.

**Ensemble weight (LightGBM):** 0.85 (val ROC-AUC 0.9987)

## Calibrated thresholds (validation)

| Model | Threshold @ 1% val FPR |
|-------|-----------------------:|
| LightGBM | 0.6779 |
| Transformer | 0.9345 |
| Ensemble | 0.6896 |

## Challenge-only detection rate

| Model | @ 0.5 | @ val-calibrated |
|-------|------:|-----------------:|
| lightgbm | 0.5381 | 0.4865 |
| transformer | 0.5835 | 0.3600 |
| ensemble | 0.5455 | 0.4803 |

## Evasive challenge TPR @ 1% FPR (mixed with test benign)

| Model | @ 0.5 threshold | @ val-calibrated threshold |
|-------|----------------:|---------------------------:|
| lightgbm | 0.4607 | 0.4607 |
| transformer | 0.2703 | 0.2703 |
| ensemble | 0.4631 | 0.4631 |
