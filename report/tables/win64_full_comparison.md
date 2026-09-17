# Win64 Model Comparison — Baselines + Transformer + Tier 1 Ablations

Same splits: **120,000 train / 40,000 test / 814 challenge (Win64)**, threshold **0.5** for detection rate. Primary seed **42**; full transformer also seeds **43, 44**.

## Primary metrics (evasive focus)

| Model | Input | Std ROC-AUC | Evasive ROC-AUC | **Challenge det.** | Evasive TPR @ 1% FPR |
|-------|-------|------------:|----------------:|-------------------:|---------------------:|
| LightGBM | Static | **0.9980** | **0.9297** | 53.8% | **0.4607** |
| CNN | Bytes | 0.9541 | 0.7967 | 33.7% | 0.0860 |
| BiLSTM | Bytes | 0.9615 | 0.8074 | 35.6% | 0.0823 |
| Transformer (static only) | Static | 0.9714 | 0.9161 | 50.9% | 0.3403 |
| Transformer (bytes only) | Bytes | 0.9583 | 0.7887 | 35.5% | 0.1400 |
| **Transformer (full, seed 42)** | Static + bytes | 0.9838 | 0.8987 | **58.4%** | 0.2703 |
| Transformer (full, mean 42–44) | Static + bytes | 0.980 | — | **55.8%** | 0.288 |

## Multi-seed (full multimodal)

| Seed | Std ROC-AUC | Challenge det. @ 0.5 | Evasive TPR @ 1% FPR |
|-----:|------------:|---------------------:|---------------------:|
| 42 | 0.9838 | 58.4% | 0.270 |
| 43 | 0.9833 | 54.3% | 0.302 |
| 44 | 0.9723 | 54.8% | 0.292 |
| Mean | 0.980 | 55.8% | 0.288 |

## Winner by metric

| Metric | Best model |
|--------|------------|
| Standard test ROC-AUC | LightGBM |
| Evasive challenge ROC-AUC | LightGBM |
| **Challenge-only detection @ 0.5** | **Transformer (full)** |
| Evasive TPR @ 1% FPR | LightGBM |
| Static-only baseline | LightGBM ≻ static-only transformer |
| Bytes-only deep models | ~tied (~34–36%); weak vs fusion |

Source: `experiments/results/*_win64_baseline/results.json`, `experiments/results/ablations/*/results.json`
