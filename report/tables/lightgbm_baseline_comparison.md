# LightGBM Baseline Comparison: Dot_Net vs Win64

Side-by-side drop-off between the **standard temporal test set** and the **evasive challenge set**.
Both runs use LightGBM on EMBER feature v3 static vectors (dim 2568), seed 42.

| Metric | Dot_Net std | Dot_Net evasive | Dot_Net drop | Win64 std | Win64 evasive | Win64 drop |
|--------|------------:|----------------:|-------------:|----------:|--------------:|-----------:|
| Accuracy | 0.9721 | 0.9559 | 0.0162 | 0.9806 | 0.9669 | 0.0136 |
| Precision | 0.9686 | 0.7468 | 0.2218 | 0.9843 | 0.5840 | **0.4003** |
| Recall | 0.9759 | 0.8432 | 0.1327 | 0.9767 | 0.5381 | **0.4386** |
| F1 | 0.9722 | 0.7921 | 0.1802 | 0.9805 | 0.5601 | **0.4204** |
| ROC-AUC | 0.9966 | 0.9717 | 0.0249 | 0.9980 | 0.9297 | **0.0682** |
| PR-AUC | 0.9967 | 0.8876 | 0.1091 | 0.9981 | 0.5602 | **0.4379** |
| TPR @ 1% FPR | 0.9189 | 0.7069 | 0.2121 | 0.9670 | 0.4607 | **0.5063** |
| Challenge-only detection | — | 84.3% | — | — | **53.8%** | — |

## Dataset sizes

| Run | Train | Test | Challenge (same file type) |
|-----|------:|-----:|---------------------------:|
| Dot_Net (laptop) | 40,000 | 15,000 | 829 |
| Win64 (Colab) | 120,000 | 40,000 | 814 |

## Takeaway for the paper

Win64 achieves near-perfect performance on the standard test set (ROC-AUC 0.998) but
**only 53.8% challenge-only detection** at threshold 0.5 — a much larger evasive-set
drop-off than Dot_Net. This is the baseline floor CNN, LSTM, and the transformer must beat.

Source configs:
- `experiments/configs/lightgbm_baseline.yaml` (Dot_Net)
- `experiments/configs/lightgbm_win64_baseline.yaml` (Win64)
