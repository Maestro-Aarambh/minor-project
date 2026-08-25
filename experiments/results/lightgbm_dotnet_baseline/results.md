# LightGBM Baseline Results (EMBER2024)

**Experiment:** `lightgbm_dotnet_baseline`  
**Model:** LightGBM on EMBER feature v3 static vectors  
**File type:** `Dot_Net`  
**Seed:** `42`  
**Generated (UTC):** `2026-08-25T21:16:00.652807+00:00`  

## Dataset sizes

- Train samples used: **40000**
- Standard test samples used: **15000** (malicious=7500, benign=7500)
- Challenge-only malicious samples: **829**
- Challenge evaluation set size (challenge + test benign): **8329**

## Main comparison table

| Metric | Standard test | Evasive challenge | Drop-off (std − evasive) |
|---|---:|---:|---:|
| Accuracy | 0.9721 | 0.9559 | 0.0162 |
| Precision | 0.9686 | 0.7468 | 0.2218 |
| Recall | 0.9759 | 0.8432 | 0.1327 |
| F1 | 0.9722 | 0.7921 | 0.1802 |
| ROC-AUC | 0.9966 | 0.9717 | 0.0249 |
| PR-AUC | 0.9967 | 0.8876 | 0.1091 |
| TPR @ 1% FPR | 0.9189 | 0.7069 | 0.2121 |
| Challenge-only detection rate | — | 0.8432 | — |

## How to read this table

The research question is not 'what accuracy does LightGBM get?', but **how much does performance drop** when moving from the temporal test set to the evasive challenge set. A larger drop-off means the model relies on patterns that do not survive AV-evasive / metamorphic samples.

## Notes

- Challenge ROC/PR metrics follow the official EMBER2024 protocol: challenge malware is mixed with test-set benign files.
- Challenge-only detection rate is measured on challenge malware alone (threshold 0.5).
- Train/test sizes may be stratified subsamples for memory limits; the challenge set is used in full. Exact subsample caps are recorded in the experiment config.

Config file: `experiments/configs/lightgbm_baseline.yaml`
