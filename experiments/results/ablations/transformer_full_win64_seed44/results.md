# Multi-Seed — Full Transformer seed 44 (EMBER2024 Win64)

**Experiment:** `transformer_full_win64_seed44`  
**Model:** Multimodal transformer (seed 44)  
**Input:** Full multimodal transformer; identical architecture to seed-42 baseline.
  
**File type:** `Win64`  
**Seed:** `44`  
**Generated (UTC):** `2026-09-02T08:42:22.618761+00:00`  

## Dataset sizes

- Train samples used: **120000**
- Standard test samples used: **40000** (malicious=20000, benign=20000)
- Challenge-only malicious samples: **814**
- Challenge evaluation set size (challenge + test benign): **20814**

## Main comparison table

| Metric | Standard test | Evasive challenge | Drop-off (std − evasive) |
|---|---:|---:|---:|
| Accuracy | 0.9495 | 0.9526 | -0.0031 |
| Precision | 0.9678 | 0.4192 | 0.5487 |
| Recall | 0.9300 | 0.5479 | 0.3820 |
| F1 | 0.9485 | 0.4750 | 0.4735 |
| ROC-AUC | 0.9723 | 0.8971 | 0.0752 |
| PR-AUC | 0.9814 | 0.4258 | 0.5556 |
| TPR @ 1% FPR | 0.8744 | 0.2924 | 0.5821 |
| Challenge-only detection rate | — | 0.5479 | — |

## How to read this table

The research question is not raw accuracy on the standard test set, but **how much does performance drop** when moving from the temporal test set to the evasive challenge set. A larger drop-off means the model relies on patterns that do not survive AV-evasive / metamorphic samples.

## Notes

- Challenge ROC/PR metrics follow the official EMBER2024 protocol: challenge malware is mixed with test-set benign files.
- Challenge-only detection rate is measured on challenge malware alone (threshold 0.5).
- Train/test sizes may be stratified subsamples for memory limits; the challenge set is used in full. Exact subsample caps are recorded in the experiment config.

Config file: `E:/minor-project/experiments/configs/ablations/transformer_full_win64_seed44.yaml`
