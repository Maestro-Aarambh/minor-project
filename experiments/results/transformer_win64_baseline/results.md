# Multimodal Transformer Results (EMBER2024)

**Experiment:** `transformer_win64_baseline`  
**Model:** Multimodal transformer (static + byte sequences)  
**Input:** EMBER static feature vector (dim ~2568, standardized on train rows only) fused with a fixed-length byte sequence (4096) expanded from the EMBER byte histogram. Fusion: [CLS] + static token + 256 byte-patch tokens through a 3-layer self-attention encoder.
  
**File type:** `Win64`  
**Seed:** `42`  
**Generated (UTC):** `2026-09-02T06:33:22.098058+00:00`  

## Dataset sizes

- Train samples used: **120000**
- Standard test samples used: **40000** (malicious=20000, benign=20000)
- Challenge-only malicious samples: **814**
- Challenge evaluation set size (challenge + test benign): **20814**

## Main comparison table

| Metric | Standard test | Evasive challenge | Drop-off (std − evasive) |
|---|---:|---:|---:|
| Accuracy | 0.9426 | 0.9515 | -0.0088 |
| Precision | 0.9648 | 0.4145 | 0.5503 |
| Recall | 0.9188 | 0.5835 | 0.3353 |
| F1 | 0.9413 | 0.4847 | 0.4566 |
| ROC-AUC | 0.9838 | 0.8987 | 0.0851 |
| PR-AUC | 0.9848 | 0.4054 | 0.5794 |
| TPR @ 1% FPR | 0.8350 | 0.2703 | 0.5648 |
| Challenge-only detection rate | — | 0.5835 | — |

## How to read this table

The research question is not raw accuracy on the standard test set, but **how much does performance drop** when moving from the temporal test set to the evasive challenge set. A larger drop-off means the model relies on patterns that do not survive AV-evasive / metamorphic samples.

## Notes

- Challenge ROC/PR metrics follow the official EMBER2024 protocol: challenge malware is mixed with test-set benign files.
- Challenge-only detection rate is measured on challenge malware alone (threshold 0.5).
- Train/test sizes may be stratified subsamples for memory limits; the challenge set is used in full. Exact subsample caps are recorded in the experiment config.

Config file: `experiments/configs/transformer_win64_baseline.yaml`
