# Classifying Metamorphic Windows PE Malware Using a Self-Attention Transformer

An empirical study that asks one precise question: **does a self-attention transformer detect evasive/metamorphic Windows malware better than CNN, LSTM, and tree-based baselines — and if so, which architectural component is responsible?**

## The research gap

Existing malware classifiers (including transformer-based ones) are almost always evaluated on standard test sets where samples are already fairly detectable. Almost none have been tested against **EMBER2024's evasive challenge set** — ~6,300 samples that initially evaded *every* antivirus engine on VirusTotal. This project measures the performance **drop-off** between the standard test set and the evasive set for all four models, then uses ablation studies to isolate *why* any difference exists.

## Dataset

**EMBER2024** (Win32/Win64 PE subset):

- Temporal train/test split (weeks 1–64)
- Evasive challenge set (~6,300 samples) — the key evaluation target

See [data/README.md](data/README.md) for how to obtain it.

## Pipeline (6 stages)

| Stage | What | Where |
|-------|------|-------|
| 1 | Data preparation: byte n-grams, opcode sequences (Capstone), EMBER static features | `src/data/` |
| 2 | Baselines: LightGBM / Random Forest, 1D CNN, LSTM/BiLSTM | `src/models/baselines/` |
| 3 | Transformer encoder: custom tokenization + positional encoding | `src/models/transformer/` |
| 4 | Evaluation: standard metrics, evasive-set drop-off, drift, adversarial robustness, efficiency | `src/evaluation/` |
| 5 | Ablation studies: attention heads, depth, input representation, positional encoding | `src/ablation/` |
| 6 | Write-up: results tables, figures, final report | `report/` |

## Build order

1. Data pipeline (Stage 1)
2. Baselines (Stage 2) — early working comparison point
3. Transformer (Stage 3)
4. Full evaluation suite (Stage 4)
5. Ablations (Stage 5) — where the real findings emerge
6. Write-up (Stage 6)

## Repository layout

```
docs/          Project summary, step-by-step coding guide, code standards
data/          raw/ (EMBER2024 downloads) and processed/ (extracted features) — gitignored
src/           All source code, one concept per module
notebooks/     Exploration and analysis notebooks
experiments/   configs/ (YAML hyperparameters) and results/ (metrics, tables)
report/        Final write-up, figures, tables
```

## Key documents

- [docs/Project_Summary.md](docs/Project_Summary.md) — full project definition and research framing
- [docs/CODING_GUIDE.md](docs/CODING_GUIDE.md) — step-by-step "how it is coded and why", one chapter per stage
- [docs/CODE_STANDARDS.md](docs/CODE_STANDARDS.md) — modularity rules every source file follows

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install "signify==0.7.1"
pip install git+https://github.com/FutureComputing4AI/EMBER2024.git
```

GPU (CUDA) is required for CNN/LSTM/transformer training; LightGBM runs on CPU.

## Run baselines

```bash
# LightGBM (static features)
python -m src.evaluation.run_lightgbm_baseline --config experiments/configs/lightgbm_win64_baseline.yaml --skip-download

# 1D CNN or BiLSTM (byte sequences — use Colab GPU for Win64)
python -m src.evaluation.run_sequence_baseline --config experiments/configs/cnn_win64_baseline.yaml --skip-download
python -m src.evaluation.run_sequence_baseline --config experiments/configs/lstm_win64_baseline.yaml --skip-download
```

## Tech stack

Python, PyTorch, scikit-learn, LightGBM, pefile, Capstone, Weights & Biases / TensorBoard.
