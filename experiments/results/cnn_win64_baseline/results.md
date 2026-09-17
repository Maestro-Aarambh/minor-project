# 1D CNN Baseline Results (EMBER2024)

**Experiment:** `cnn_win64_baseline`  
**Model:** 1D CNN on histogram-derived byte sequences  
**Input:** Fixed-length byte sequence (4096) expanded from EMBER byte histogram. EMBER2024 does not ship raw PE bytes; see docs/CODING_GUIDE.md.
  
**File type:** `Win64`  
**Seed:** `42`  
**Generated (UTC):** `2026-09-01T19:27:37.736307+00:00`  

## Dataset sizes

- Train samples used: **120000**
- Standard test samples used: **40000** (malicious=20000, benign=20000)
- Challenge-only malicious samples: **814**
- Challenge evaluation set size (challenge + test benign): **20814**

## Main comparison table

| Metric | Standard test | Evasive challenge | Drop-off (std − evasive) |
|---|---:|---:|---:|
| Accuracy | 0.8857 | 0.8988 | -0.0131 |
| Precision | 0.9156 | 0.1489 | 0.7667 |
| Recall | 0.8498 | 0.3366 | 0.5131 |
| F1 | 0.8815 | 0.2065 | 0.6750 |
| ROC-AUC | 0.9541 | 0.7967 | 0.1573 |
| PR-AUC | 0.9537 | 0.1632 | 0.7905 |
| TPR @ 1% FPR | 0.4480 | 0.0860 | 0.3620 |
| Challenge-only detection rate | — | 0.3366 | — |

## How to read this table

The research question is not raw accuracy on the standard test set, but **how much does performance drop** when moving from the temporal test set to the evasive challenge set. A larger drop-off means the model relies on patterns that do not survive AV-evasive / metamorphic samples.

## Notes

- Challenge ROC/PR metrics follow the official EMBER2024 protocol: challenge malware is mixed with test-set benign files.
- Challenge-only detection rate is measured on challenge malware alone (threshold 0.5).
- Train/test sizes may be stratified subsamples for memory limits; the challenge set is used in full. Exact subsample caps are recorded in the experiment config.

Config file: `experiments/configs/cnn_win64_baseline.yaml`
