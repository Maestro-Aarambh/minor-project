# Multi-Seed — Full Transformer seed 43 (EMBER2024 Win64)

**Experiment:** `transformer_full_win64_seed43`  
**Model:** Multimodal transformer (seed 43)  
**Input:** Full multimodal transformer; identical architecture to seed-42 baseline.
  
**File type:** `Win64`  
**Seed:** `43`  
**Generated (UTC):** `2026-09-02T08:04:01.869509+00:00`  

## Dataset sizes

- Train samples used: **120000**
- Standard test samples used: **40000** (malicious=20000, benign=20000)
- Challenge-only malicious samples: **814**
- Challenge evaluation set size (challenge + test benign): **20814**

## Main comparison table

| Metric | Standard test | Evasive challenge | Drop-off (std − evasive) |
|---|---:|---:|---:|
| Accuracy | 0.9466 | 0.9559 | -0.0093 |
| Precision | 0.9712 | 0.4478 | 0.5234 |
| Recall | 0.9205 | 0.5430 | 0.3776 |
| F1 | 0.9452 | 0.4908 | 0.4544 |
| ROC-AUC | 0.9833 | 0.9026 | 0.0808 |
| PR-AUC | 0.9861 | 0.4397 | 0.5464 |
| TPR @ 1% FPR | 0.8717 | 0.3022 | 0.5695 |
| Challenge-only detection rate | — | 0.5430 | — |

## How to read this table

The research question is not raw accuracy on the standard test set, but **how much does performance drop** when moving from the temporal test set to the evasive challenge set. A larger drop-off means the model relies on patterns that do not survive AV-evasive / metamorphic samples.

## Notes

- Challenge ROC/PR metrics follow the official EMBER2024 protocol: challenge malware is mixed with test-set benign files.
- Challenge-only detection rate is measured on challenge malware alone (threshold 0.5).
- Train/test sizes may be stratified subsamples for memory limits; the challenge set is used in full. Exact subsample caps are recorded in the experiment config.

Config file: `E:/minor-project/experiments/configs/ablations/transformer_full_win64_seed43.yaml`
