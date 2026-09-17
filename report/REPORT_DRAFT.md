# Classifying Metamorphic Windows PE Malware Using a Multimodal Self-Attention Transformer

**First draft report — EMBER2024 Win64 experiments**  
**Author:** [Your name]  
**Date:** September 2026  
**Repository:** `minor-project`  
**Primary seed:** 42 (all baselines)  
**Multi-seed (full transformer):** 42, 43, 44

---

## Abstract

We study whether a self-attention transformer can detect **AV-evasive** Windows PE malware better than classical and deep learning baselines on **EMBER2024**. Using identical Win64 splits (120,000 train / 40,000 test / 814 evasive challenge samples), we compare LightGBM on EMBER static features, a 1D CNN, a BiLSTM, and a **multimodal transformer** that fuses static PE features with histogram-derived byte sequences via self-attention. On the **standard temporal test set**, LightGBM achieves the highest ROC-AUC (0.998). On the **evasive challenge set**—samples that initially evaded every antivirus engine—the multimodal transformer achieves the best **challenge-only detection rate (58.4% at threshold 0.5; mean 55.8% over seeds 42–44)**, outperforming LightGBM (53.8%), BiLSTM (35.6%), and CNN (33.7%). Controlled **modality ablations** show that fusion is required *within* the transformer: static-only attention (50.9%) underperforms LightGBM on the same static features, and bytes-only attention (35.5%) matches CNN/BiLSTM. LightGBM remains stronger on ROC-AUC and TPR @ 1% FPR. We report validation-calibrated operating points and limitations (histogram proxy, no raw bytes/opcodes).

---

## 1. Introduction

### 1.1 Problem

Metamorphic and evasive malware deliberately alter their surface appearance to evade signature-based and many machine-learning detectors. Benchmarks that evaluate only on **easy** temporal test splits may overstate real-world robustness.

### 1.2 Research question

> **Does a self-attention transformer detect evasive Win64 malware better than LightGBM, CNN, and LSTM—and if so, does multimodal fusion (static + byte sequence) explain the gain?**

The central metric is **performance drop-off** when moving from the standard EMBER2024 test set to the **evasive challenge set**, not raw accuracy on easy malware.

### 1.3 Contributions

1. End-to-end reproducible pipeline: EMBER2024 JSONL → static / byte / multimodal caches → train → standard + evasive evaluation.
2. Fair comparison on **identical Win64 subsamples** (same seed, file caps, challenge filter).
3. Empirical finding: **multimodal transformer beats all baselines on challenge-only detection @ 0.5** (58.4% seed 42; mean 55.8% over seeds 42–44 vs LightGBM 53.8%).
4. **Tier 1 modality ablations:** static-only and bytes-only transformers isolate fusion; LightGBM remains the strong static-only tree baseline (not a “missing” static run).
5. Multi-seed replication and val-calibrated operating-point / ensemble analysis (no challenge tuning).

---

## 2. Dataset

### 2.1 EMBER2024

- **Source:** FutureComputing4AI EMBER2024 (HuggingFace / `thrember`).
- **Subset used:** **Win64** PE features (JSONL records, not raw binaries).
- **Splits:**
  - **Train:** weekly `*_Win64_train.jsonl` (16 files subsampled).
  - **Test:** weekly `*_Win64_test.jsonl` (8 files subsampled).
  - **Challenge:** evasive malware JSONL, filtered to Win64 → **814 samples** (all malicious).

### 2.2 Subsample caps (compute limits)

| Split | Cap | Malicious | Benign |
|-------|-----|-----------|--------|
| Train | 120,000 | 60,000 | 60,000 |
| Test | 40,000 | 20,000 | 20,000 |
| Challenge | Full (Win64 filter) | 814 | 0 |

Stratified subsampling with **seed 42** preserves class balance. The challenge set is **never subsampled**.

### 2.3 Challenge evaluation protocol

Following official EMBER2024 practice:

- **Challenge ROC/PR:** challenge malware concatenated with **test-set benign** samples (20,000) so both classes exist.
- **Challenge-only detection rate:** fraction of the 814 challenge malware scored as malicious at **threshold 0.5**.

---

## 3. Feature engineering

All models consume the **same underlying JSONL files** but different numeric views.

### 3.1 Static features (LightGBM, transformer static branch)

- **Extractor:** `thrember.PEFeatureExtractor` (EMBER feature v3, **dim 2568**).
- **Content:** PE headers, sections, imports, byte histogram/entropy summaries, strings, etc.
- **Cache:** `data/processed/static_win64/`.

### 3.2 Histogram-derived byte sequences (CNN, LSTM, transformer byte branch)

EMBER2024 does **not** ship raw PE byte streams. We expand each file's **256-bin byte histogram** deterministically into a fixed **4096-length integer sequence** (bytes 0–255 repeated in proportion to counts).

- **Cache:** `data/processed/bytes_win64/`.
- **Limitation:** This is **not** MalConv-style raw byte order; it encodes **byte composition** only. Document this in any publication.

### 3.3 Multimodal alignment

Static and byte caches are built with the **same** `seed`, `train_max` / `test_max`, and `train_max_files` / `test_max_files`. Row *i* in static matches row *i* in bytes; labels are verified element-wise before training (`src/data/multimodal_features.py`).

### 3.4 What we did not use

- **Raw PE binaries** (not in public EMBER JSONL).
- **Opcode sequences** (Capstone disassembly planned but not implemented—requires binaries).

---

## 4. Models

### 4.1 LightGBM (static baseline)

- Gradient-boosted trees on 2568-D static vectors.
- Strong classical EMBER baseline; trains in ~10 minutes on Win64 (CPU).

### 4.2 1D CNN (local patterns)

- Embedding + stacked Conv1d (kernel 7) + global max pool.
- Input: `(4096,)` byte sequence only.

### 4.3 BiLSTM (sequential patterns)

- 2-layer bidirectional LSTM + max pool.
- Same byte input as CNN; batch size 32 on 4 GB GPU (128 OOM).

### 4.4 Multimodal transformer (proposed)

**Architecture:**

```
[CLS] + [STATIC token] + [256 byte-patch tokens] → 3-layer TransformerEncoder → CLS head
```

| Component | Setting |
|-----------|---------|
| Static input | 2568-D, standardized (train mean/std only) |
| Byte input | 4096 bytes → patches of 16 → 256 tokens |
| `d_model` | 128 |
| Heads | 4 |
| Layers | 3 |
| FFN dim | 256 |
| Dropout | 0.2 |
| Parameters | ~806,000 |

**Training controls (leakage / overfit):**

- 10% stratified **validation split from train only** (never test/challenge).
- Class-weighted BCE, AdamW, weight decay, gradient clipping.
- Early stopping (patience 3) on val loss; **best checkpoint restored**.
- Test and challenge used **once** for final metrics.

---

## 5. Experimental setup

| Setting | Value |
|---------|--------|
| Hardware (deep models) | NVIDIA RTX 2050 (4 GB VRAM), CUDA 12.6 |
| Framework | PyTorch 2.x, LightGBM, scikit-learn |
| Decision threshold | 0.5 (fixed; calibration left as future work) |
| Leakage checks | Separate EMBER train/test files; val from train only; scaler fit on train rows only |

### 5.1 Warm-up runs (pipeline validation)

| Run | File type | Train | Test | Challenge det. @ 0.5 |
|-----|-----------|------:|-----:|---------------------:|
| LightGBM | Dot_Net | 40k | 15k | 84.3% |
| CNN smoke | Dot_Net | 800 | 400 | (smoke test only) |

Win64 is the **primary** evaluation target (harder evasive drop-off than Dot_Net).

---

## 6. Main results (Win64)

### 6.1 Summary table

| Model | Input | Std ROC-AUC | Evasive ROC-AUC | ROC drop | Challenge det. @ 0.5 | Std TPR @ 1% FPR | Evasive TPR @ 1% FPR |
|-------|-------|------------:|----------------:|---------:|---------------------:|-----------------:|---------------------:|
| **LightGBM** | Static (~2568-D) | **0.9980** | **0.9297** | **0.0682** | 53.8% | **0.9670** | **0.4607** |
| **CNN** | Bytes (4096) | 0.9541 | 0.7967 | 0.1573 | 33.7% | 0.4480 | 0.0860 |
| **BiLSTM** | Bytes (4096) | 0.9615 | 0.8074 | 0.1542 | 35.6% | 0.5980 | 0.0823 |
| **Multimodal transformer** | Static + bytes | 0.9838 | 0.8987 | 0.0851 | **58.4%** | 0.8350 | 0.2703 |

*Challenge detection = challenge-only malicious, threshold 0.5. Drop-off = standard minus evasive (ROC-AUC).*

### 6.2 Full metrics — standard test

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|---------:|----------:|-------:|---:|--------:|-------:|
| LightGBM | 0.9806 | 0.9843 | 0.9767 | 0.9805 | **0.9980** | **0.9981** |
| CNN | 0.8857 | 0.9156 | 0.8498 | 0.8815 | 0.9541 | 0.9537 |
| BiLSTM | 0.8968 | 0.9220 | 0.8669 | 0.8936 | 0.9615 | 0.9636 |
| Transformer | 0.9426 | 0.9648 | 0.9188 | 0.9413 | 0.9838 | 0.9848 |

### 6.3 Full metrics — evasive challenge (mixed eval)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|---------:|----------:|-------:|---:|--------:|-------:|
| LightGBM | 0.9669 | 0.5840 | 0.5381 | 0.5601 | **0.9297** | 0.5602 |
| CNN | 0.8988 | 0.1489 | 0.3366 | 0.2065 | 0.7967 | 0.1632 |
| BiLSTM | 0.9043 | 0.1651 | 0.3563 | 0.2256 | 0.8074 | 0.1567 |
| Transformer | 0.9515 | **0.4145** | **0.5835** | **0.4847** | 0.8987 | 0.4054 |

---

## 7. Findings and discussion

### 7.1 Answer to the research question

**Yes**, on our Win64 setup the **multimodal transformer achieves the highest evasive challenge detection (58.4%)**, beating LightGBM (+4.6 pp), BiLSTM (+22.8 pp), and CNN (+24.7 pp) at threshold 0.5.

The gain aligns with the hypothesis that **combining EMBER static structure with sequential byte-composition patterns via self-attention** helps on malware engineered to evade AV—provided the model sees **both** representations (static alone in trees already strong; bytes alone weak).

### 7.2 Standard test vs evasive challenge

- **LightGBM** dominates the **standard test** (ROC-AUC 0.998) and has the **smallest ROC drop** (0.068)—excellent on temporal test malware.
- **Win64 evasive challenge is much harder:** even LightGBM detects only **~54%** at 0.5.
- **CNN/LSTM on histogram bytes alone fail** on evasive samples (~34–36%), showing the byte proxy without static PE features is insufficient.
- **Transformer** trades a small amount of standard-test ROC (0.984 vs 0.998) for the **best evasive recall** at 0.5.

### 7.3 Operating-point trade-off (TPR @ 1% FPR)

At a **strict 1% false-positive rate** on the evasive mixed eval, **LightGBM remains best** (TPR 0.461 vs transformer 0.270). This suggests:

- The transformer improves **balanced challenge detection @ 0.5**, not uniformly across all operating points.
- **Threshold calibration** on a validation set (targeting 1% FPR) is important future work and may change the ranking.

### 7.4 Why accuracy on evasive can exceed standard test

Several models show **higher accuracy on the evasive mixed set** than on standard test. This is **misleading**: the mixed challenge set is ~96% benign (20k benign + 814 malware). High accuracy often means predicting benign. **Challenge-only detection rate** and **ROC-AUC** are the metrics to trust.

### 7.5 Training efficiency

| Model | Train time (Win64) | Notes |
|-------|-------------------:|-------|
| LightGBM | ~587 s (~10 min) | CPU |
| CNN | ~2,870 s (~48 min) | GPU, batch 128 |
| BiLSTM | ~33,529 s (~9.3 hr) | GPU, batch 32 (OOM at 128) |
| Transformer | ~2,681 s (~45 min) | GPU, batch 32; early stop epoch 13 |

The transformer is **~7× faster than BiLSTM** on the same hardware while delivering the best evasive detection. Patch-based attention (256 tokens) vs full-length LSTM (4096 steps) explains much of the gap.

---

## 8. Ablation analysis

Tier 1 formal ablations are **complete**. Artifacts: [`report/tables/ablations/tier1_ablation_summary.md`](tables/ablations/tier1_ablation_summary.md), [`tier1_operating_points.md`](tables/ablations/tier1_operating_points.md). Runbook: [`docs/TIER1_STUDY.md`](../docs/TIER1_STUDY.md).

### 8.1 Modality ablation (controlled transformer branches)

Same architecture, training budget, and splits; only the token set changes (`modality: full | static_only | bytes_only`). LightGBM is the **tree** static-only baseline on the same EMBER vectors (not an ablation of the transformer).

| Configuration | Model | Std ROC-AUC | Challenge det. @ 0.5 | Evasive TPR @ 1% FPR |
|---------------|-------|------------:|---------------------:|---------------------:|
| Static only (trees) | LightGBM | **0.9980** | 53.8% | **0.461** |
| Static only (attention) | Transformer | 0.9714 | 50.9% | 0.340 |
| Bytes only (attention) | Transformer | 0.9583 | 35.5% | 0.140 |
| Bytes only (CNN) | CNN | 0.9541 | 33.7% | 0.086 |
| Bytes only (LSTM) | BiLSTM | 0.9615 | 35.6% | 0.082 |
| **Static + bytes (fusion)** | Transformer | 0.9838 | **58.4%** | 0.270 |

**Findings**

1. **Fusion is required within the transformer.** Full multimodal challenge detection (58.4%) exceeds static-only attention (50.9%, **+7.5 pp**) and bytes-only attention (35.5%, **+22.9 pp**).
2. **Static PE features remain strong alone.** LightGBM on static features (53.8%) beats the static-only transformer (50.9%): same input, trees outperform a single static token through self-attention.
3. **Bytes-only attention ≈ CNN/BiLSTM (~35%).** The weak signal is the histogram-derived byte proxy, not the choice of sequence architecture.
4. **“Fusion necessary” does not mean “static-only methods fail.”** It means ablating either branch *of the multimodal model* hurts that model; LightGBM is intentionally the strong static-only classical baseline.

### 8.2 Multi-seed replication (full multimodal)

| Seed | Std ROC-AUC | Challenge det. @ 0.5 | Evasive TPR @ 1% FPR | Train (s) |
|-----:|------------:|---------------------:|---------------------:|----------:|
| 42 | 0.9838 | **58.4%** | 0.270 | 2681 |
| 43 | 0.9833 | 54.3% | 0.302 | 1510 |
| 44 | 0.9723 | 54.8% | 0.292 | 2222 |
| **Mean** | **0.980** | **55.8%** | **0.288** | — |

Seed 42 is the best challenge run. The mean (**55.8%**) still exceeds LightGBM at threshold 0.5 (**53.8%**, +2.0 pp), but the margin is smaller than the single-seed headline (+4.6 pp). LightGBM remains clearly better on **TPR @ 1% FPR** (~0.46 vs ~0.29).

### 8.3 Operating points and ensemble (validation-only tuning)

Thresholds and ensemble weights were tuned on a **10% stratified holdout from train** (seed 42). Test and challenge were never used for tuning.

| Model | Threshold @ ~1% val FPR | Challenge det @ 0.5 | Challenge det @ calibrated | Evasive TPR @ 1% FPR |
|-------|------------------------:|--------------------:|---------------------------:|---------------------:|
| LightGBM | 0.678 | 53.8% | 48.7% | **0.461** |
| Transformer (full) | 0.935 | **58.4%** | 36.0% | 0.270 |
| Ensemble (85% LGBM + 15% TF) | 0.690 | 54.6% | 48.0% | **0.463** |

Val calibration raises the transformer threshold sharply and **reduces** challenge detection at that operating point. For publication we report **both** fixed 0.5 and low-FPR metrics: transformer leads at 0.5; LightGBM (and the LGBM-heavy ensemble) lead at ~1% FPR.

### 8.4 Architecture on same byte input (implicit)

| Pair | Δ Challenge det. | Δ Evasive ROC-AUC |
|------|-----------------:|------------------:|
| BiLSTM − CNN | +1.9 pp | +0.011 |
| Bytes-only transformer − BiLSTM | −0.1 pp | −0.019 |
| Full transformer − BiLSTM | +22.8 pp | +0.091 |

Sequential modeling adds little on histogram bytes alone; **multimodal fusion** drives the large gain over sequence-only models.

### 8.5 Tier 2 ablations (future)

| Axis | Variants | Purpose |
|------|----------|---------|
| Attention heads | 2, 4, 8 | Capacity vs overfit |
| Encoder depth | 1, 3, 6 layers | Long-range vs compute |
| Patch size | 8, 16, 32 | Token count vs local detail |
| Positional encoding | Sinusoidal vs learned | Structural inductive bias |
| Opcode tokens | Capstone on binaries | Behavior vs surface bytes |

---

## 9. Limitations

1. **Histogram byte proxy**, not raw file bytes—limits comparison to MalConv-style literature.
2. **No opcode / disassembly branch**—metamorphism often preserves logic while changing bytes.
3. **Subsampled train/test** (120k/40k)—full EMBER Win64 not used due to RAM/time.
4. **Three seeds for full transformer only** (42–44); modality ablations and baselines remain seed 42.
5. **Operating-point trade-off:** transformer leads at threshold 0.5; LightGBM leads at ~1% FPR / val-calibrated thresholds.
6. **Static scaler** uses float64 mean/std on train rows; monitor numerical warnings on other hardware.
7. **Single GPU, one machine**—no distributed training or Tier 2 hyperparameter sweeps yet.

---

## 10. Threats to validity

| Threat | Mitigation |
|--------|------------|
| Train/test leakage | EMBER temporal file split; test never in fit(); val from train only |
| Challenge tuning | Challenge evaluated once; no hyperparameter search on challenge |
| Cache misalignment | Label equality check between static and byte caches |
| Overfitting | Early stopping, dropout, weight decay, best-val checkpoint |

---

## 11. Conclusion

We built a reproducible malware classification pipeline on **EMBER2024 Win64** comparing LightGBM, CNN, BiLSTM, and a **multimodal self-attention transformer** on identical splits. LightGBM achieves near-perfect **standard-test** performance, the smallest **ROC-AUC drop**, and the best **TPR @ 1% FPR**. Sequence-only deep models on histogram-derived bytes detect only ~34–36% of evasive challenge malware. The **fused multimodal transformer** achieves the best **challenge-only detection at threshold 0.5** (**58.4%** on seed 42; **mean 55.8%** over seeds 42–44 vs LightGBM **53.8%**).

Tier 1 ablations clarify *why*: a **static-only transformer** (50.9%) underperforms LightGBM on the same static features, and a **bytes-only transformer** (35.5%) matches CNN/BiLSTM—so fusion is necessary *within* the attention model, while strong static-only tree models remain competitive. Val-calibrated thresholds favor LightGBM-style operating points; we therefore report both fixed-threshold and low-FPR metrics.

Future work: Tier 2 architecture sweeps (depth, heads, patch size), richer byte/opcode inputs when binaries are available, and stronger ensembles.

---

## 12. Reproducibility

### 12.1 Config files

| Experiment | Config |
|------------|--------|
| LightGBM Win64 | `experiments/configs/lightgbm_win64_baseline.yaml` |
| CNN Win64 | `experiments/configs/cnn_win64_baseline.yaml` |
| BiLSTM Win64 | `experiments/configs/lstm_win64_baseline.yaml` |
| Transformer Win64 | `experiments/configs/transformer_win64_baseline.yaml` |
| Vectorize bytes | `experiments/configs/vectorize_bytes_win64.yaml` |
| Tier 1 manifest | `experiments/configs/tier1_ablation_manifest.yaml` |
| Operating points | `experiments/configs/analyze_operating_points.yaml` |

### 12.2 Commands (Win64, caches present)

```bash
python -m src.evaluation.run_lightgbm_baseline --config experiments/configs/lightgbm_win64_baseline.yaml --skip-download --skip-vectorize
python -m src.evaluation.run_sequence_baseline --config experiments/configs/cnn_win64_baseline.yaml --skip-download --skip-vectorize
python -m src.evaluation.run_sequence_baseline --config experiments/configs/lstm_win64_baseline.yaml --skip-download --skip-vectorize
python -m src.evaluation.run_transformer_baseline --config experiments/configs/transformer_win64_baseline.yaml --skip-download --skip-vectorize

# Tier 1 ablations + multi-seed + operating points
python -m src.ablation.run_tier1 --skip-download --skip-vectorize
python -m src.evaluation.analyze_operating_points --skip-download --skip-vectorize
python -m src.ablation.run_tier1 --summarize
```

### 12.3 Result artifacts

| Model | Results |
|-------|---------|
| LightGBM | `experiments/results/lightgbm_win64_baseline/` |
| CNN | `experiments/results/cnn_win64_baseline/` |
| BiLSTM | `experiments/results/lstm_win64_baseline/` |
| Transformer (seed 42) | `experiments/results/transformer_win64_baseline/` |
| Static-only / bytes-only / seeds 43–44 | `experiments/results/ablations/` |
| Operating points | `experiments/results/tier1_operating_points/` |
| Tables | `report/tables/*_win64_baseline_results.md`, `report/tables/ablations/` |

---

## Appendix A — Dot_Net vs Win64 (LightGBM only)

Win64 is a harder evasive target than Dot_Net for static features:

| File type | Std ROC-AUC | Challenge det. @ 0.5 |
|-----------|------------:|---------------------:|
| Dot_Net | 0.9966 | 84.3% |
| Win64 | 0.9980 | 53.8% |

Near-perfect standard-test scores on both; **Win64 challenge detection drops much more**, motivating focus on Win64 for evasive evaluation.

---

## Appendix B — Transformer training curve

| Epoch | Train loss | Val loss |
|------:|-----------:|---------:|
| 1 | 0.1449 | 0.1109 |
| 4 | 0.0721 | 0.0848 |
| 7 | 0.0535 | 0.0789 |
| 10 | 0.0416 | **0.0789** (best checkpoint) |
| 13 | 0.0361 | 0.0816 → early stop |

Best validation loss at epoch 10; training stopped at epoch 13 (patience 3).

---

## Appendix C — References (placeholder)

1. EMBER2024 dataset and `thrember` tooling — FutureComputing4AI, 2024/2025.
2. EMBER feature v3 — Anderson & Roth, *ARXIV* (original EMBER paper family).
3. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
4. Raff et al., MalConv — raw byte CNN baseline (contrast with histogram proxy).

---

*End of first draft. Update author name, institution, and references before submission.*
