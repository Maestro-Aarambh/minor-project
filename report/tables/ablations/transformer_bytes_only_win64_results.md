# Tier 1 Ablation — Bytes-Only Transformer (EMBER2024 Win64)

**Experiment:** `transformer_bytes_only_win64`  
**Model:** Transformer (bytes only)  
**Input:** Histogram-derived byte sequence as 256 patch tokens + CLS through self-attention. Static EMBER feature branch disabled.
  
**File type:** `Win64`  
**Seed:** `42`  
**Generated (UTC):** `2026-09-02T07:37:50.984642+00:00`  

## Dataset sizes

- Train samples used: **120000**
- Standard test samples used: **40000** (malicious=20000, benign=20000)
- Challenge-only malicious samples: **814**
- Challenge evaluation set size (challenge + test benign): **20814**

## Main comparison table

| Metric | Standard test | Evasive challenge | Drop-off (std − evasive) |
|---|---:|---:|---:|
| Accuracy | 0.8928 | 0.8977 | -0.0049 |
| Precision | 0.9152 | 0.1527 | 0.7625 |
| Recall | 0.8658 | 0.3550 | 0.5107 |
| F1 | 0.8898 | 0.2135 | 0.6763 |
| ROC-AUC | 0.9583 | 0.7887 | 0.1695 |
| PR-AUC | 0.9605 | 0.1810 | 0.7795 |
| TPR @ 1% FPR | 0.5776 | 0.1400 | 0.4376 |
| Challenge-only detection rate | — | 0.3550 | — |

## How to read this table

The research question is not raw accuracy on the standard test set, but **how much does performance drop** when moving from the temporal test set to the evasive challenge set. A larger drop-off means the model relies on patterns that do not survive AV-evasive / metamorphic samples.

## Notes

- Challenge ROC/PR metrics follow the official EMBER2024 protocol: challenge malware is mixed with test-set benign files.
- Challenge-only detection rate is measured on challenge malware alone (threshold 0.5).
- Train/test sizes may be stratified subsamples for memory limits; the challenge set is used in full. Exact subsample caps are recorded in the experiment config.

Config file: `E:/minor-project/experiments/configs/ablations/transformer_bytes_only_win64.yaml`
