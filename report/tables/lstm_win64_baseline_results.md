# BiLSTM Baseline Results (EMBER2024)

**Experiment:** `lstm_win64_baseline`  
**Model:** BiLSTM on histogram-derived byte sequences  
**Input:** Fixed-length byte sequence (4096) expanded from EMBER byte histogram. EMBER2024 does not ship raw PE bytes; see docs/CODING_GUIDE.md.
  
**File type:** `Win64`  
**Seed:** `42`  
**Generated (UTC):** `2026-09-02T05:01:37.889233+00:00`  

## Dataset sizes

- Train samples used: **120000**
- Standard test samples used: **40000** (malicious=20000, benign=20000)
- Challenge-only malicious samples: **814**
- Challenge evaluation set size (challenge + test benign): **20814**

## Main comparison table

| Metric | Standard test | Evasive challenge | Drop-off (std − evasive) |
|---|---:|---:|---:|
| Accuracy | 0.8968 | 0.9043 | -0.0076 |
| Precision | 0.9220 | 0.1651 | 0.7569 |
| Recall | 0.8669 | 0.3563 | 0.5106 |
| F1 | 0.8936 | 0.2256 | 0.6680 |
| ROC-AUC | 0.9615 | 0.8074 | 0.1542 |
| PR-AUC | 0.9636 | 0.1567 | 0.8069 |
| TPR @ 1% FPR | 0.5980 | 0.0823 | 0.5157 |
| Challenge-only detection rate | — | 0.3563 | — |

## How to read this table

The research question is not raw accuracy on the standard test set, but **how much does performance drop** when moving from the temporal test set to the evasive challenge set. A larger drop-off means the model relies on patterns that do not survive AV-evasive / metamorphic samples.

## Notes

- Challenge ROC/PR metrics follow the official EMBER2024 protocol: challenge malware is mixed with test-set benign files.
- Challenge-only detection rate is measured on challenge malware alone (threshold 0.5).
- Train/test sizes may be stratified subsamples for memory limits; the challenge set is used in full. Exact subsample caps are recorded in the experiment config.

Config file: `experiments/configs/lstm_win64_baseline.yaml`
