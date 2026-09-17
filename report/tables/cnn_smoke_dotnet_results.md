# CNN Smoke Test Results

**Experiment:** `cnn_smoke_dotnet`  
**Model:** 1D CNN (smoke test)  
**Input:** Histogram-derived byte sequence (1024), Dot_Net subset  
**File type:** `Dot_Net`  
**Seed:** `42`  
**Generated (UTC):** `2026-08-30T12:57:42.210889+00:00`  

## Dataset sizes

- Train samples used: **800**
- Standard test samples used: **400** (malicious=200, benign=200)
- Challenge-only malicious samples: **829**
- Challenge evaluation set size (challenge + test benign): **1029**

## Main comparison table

| Metric | Standard test | Evasive challenge | Drop-off (std − evasive) |
|---|---:|---:|---:|
| Accuracy | 0.7600 | 0.3032 | 0.4568 |
| Precision | 0.8250 | 0.8333 | -0.0083 |
| Recall | 0.6600 | 0.1689 | 0.4911 |
| F1 | 0.7333 | 0.2808 | 0.4525 |
| ROC-AUC | 0.8314 | 0.6011 | 0.2302 |
| PR-AUC | 0.8378 | 0.8358 | 0.0020 |
| TPR @ 1% FPR | 0.1400 | 0.0133 | 0.1267 |
| Challenge-only detection rate | — | 0.1689 | — |

## How to read this table

The research question is not raw accuracy on the standard test set, but **how much does performance drop** when moving from the temporal test set to the evasive challenge set. A larger drop-off means the model relies on patterns that do not survive AV-evasive / metamorphic samples.

## Notes

- Challenge ROC/PR metrics follow the official EMBER2024 protocol: challenge malware is mixed with test-set benign files.
- Challenge-only detection rate is measured on challenge malware alone (threshold 0.5).
- Train/test sizes may be stratified subsamples for memory limits; the challenge set is used in full. Exact subsample caps are recorded in the experiment config.

Config file: `experiments/configs/cnn_smoke_dotnet.yaml`
