# LightGBM Baseline Results (EMBER2024)

**Experiment:** `lightgbm_win64_baseline`  
**Model:** LightGBM on EMBER feature v3 static vectors  
**File type:** `Win64`  
**Seed:** `42`  
**Generated (UTC):** `2026-08-30T12:19:36.599769+00:00`  

## Dataset sizes

- Train samples used: **120000**
- Standard test samples used: **40000** (malicious=20000, benign=20000)
- Challenge-only malicious samples: **814**
- Challenge evaluation set size (challenge + test benign): **20814**

## Main comparison table

| Metric | Standard test | Evasive challenge | Drop-off (std − evasive) |
|---|---:|---:|---:|
| Accuracy | 0.9806 | 0.9669 | 0.0136 |
| Precision | 0.9843 | 0.5840 | 0.4003 |
| Recall | 0.9767 | 0.5381 | 0.4386 |
| F1 | 0.9805 | 0.5601 | 0.4204 |
| ROC-AUC | 0.9980 | 0.9297 | 0.0682 |
| PR-AUC | 0.9981 | 0.5602 | 0.4379 |
| TPR @ 1% FPR | 0.9670 | 0.4607 | 0.5063 |
| Challenge-only detection rate | — | 0.5381 | — |

## How to read this table

The research question is not 'what accuracy does LightGBM get?', but **how much does performance drop** when moving from the temporal test set to the evasive challenge set. A larger drop-off means the model relies on patterns that do not survive AV-evasive / metamorphic samples.

## Notes

- Challenge ROC/PR metrics follow the official EMBER2024 protocol: challenge malware is mixed with test-set benign files.
- Challenge-only detection rate is measured on challenge malware alone (threshold 0.5).
- Train/test sizes may be stratified subsamples for memory limits; the challenge set is used in full. Exact subsample caps are recorded in the experiment config.

Config file: `experiments/configs/lightgbm_win64_baseline.yaml`
