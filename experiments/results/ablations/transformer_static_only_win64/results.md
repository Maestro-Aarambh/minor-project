# Tier 1 Ablation — Static-Only Transformer (EMBER2024 Win64)

**Experiment:** `transformer_static_only_win64`  
**Model:** Transformer (static only)  
**Input:** EMBER static vector (dim ~2568) as a single token + CLS through self-attention. Byte patch branch disabled.
  
**File type:** `Win64`  
**Seed:** `42`  
**Generated (UTC):** `2026-09-02T08:54:38.402967+00:00`  

## Dataset sizes

- Train samples used: **120000**
- Standard test samples used: **40000** (malicious=20000, benign=20000)
- Challenge-only malicious samples: **814**
- Challenge evaluation set size (challenge + test benign): **20814**

## Main comparison table

| Metric | Standard test | Evasive challenge | Drop-off (std − evasive) |
|---|---:|---:|---:|
| Accuracy | 0.9405 | 0.9584 | -0.0179 |
| Precision | 0.9749 | 0.4705 | 0.5044 |
| Recall | 0.9042 | 0.5086 | 0.3956 |
| F1 | 0.9382 | 0.4888 | 0.4494 |
| ROC-AUC | 0.9714 | 0.9161 | 0.0553 |
| PR-AUC | 0.9791 | 0.4558 | 0.5233 |
| TPR @ 1% FPR | 0.8633 | 0.3403 | 0.5230 |
| Challenge-only detection rate | — | 0.5086 | — |

## How to read this table

The research question is not raw accuracy on the standard test set, but **how much does performance drop** when moving from the temporal test set to the evasive challenge set. A larger drop-off means the model relies on patterns that do not survive AV-evasive / metamorphic samples.

## Notes

- Challenge ROC/PR metrics follow the official EMBER2024 protocol: challenge malware is mixed with test-set benign files.
- Challenge-only detection rate is measured on challenge malware alone (threshold 0.5).
- Train/test sizes may be stratified subsamples for memory limits; the challenge set is used in full. Exact subsample caps are recorded in the experiment config.

Config file: `experiments/configs/ablations/transformer_static_only_win64.yaml`
