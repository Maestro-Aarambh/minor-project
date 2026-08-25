# Code Standards

Rules every source file in this project follows. The goal: any single module
can be explained on one presentation slide, and every result in the paper can
be traced back to an exact configuration.

## 1. One concept per module

Each file does exactly one job and is named after it:

- `byte_tokenizer.py` — turns raw bytes into token sequences, nothing else
- `positional_encoding.py` — positional encoding variants, nothing else
- `evasive_eval.py` — evasive-set evaluation, nothing else

If a file needs the word "and" to describe what it does, split it.

## 2. Config-driven, never hardcoded

All hyperparameters (sequence length, learning rate, number of attention
heads, encoder depth, batch size, ...) live in YAML files under
`experiments/configs/`. Code reads config; code never contains magic numbers.

Why: every table in the paper cites the config file that produced it, so
results are reproducible and the ablation study is just "same code,
different config".

## 3. Common model interface

All four models (LightGBM, CNN, LSTM, Transformer) implement the same API:

```python
class BaseClassifier:
    def fit(self, X, y): ...
    def predict_proba(self, X): ...
    def save(self, path): ...
    def load(self, path): ...
```

Why: the evaluation code treats every model identically, which makes the
"fair comparison" claim in the paper defensible — no model gets special
treatment in the harness.

## 4. Docstrings explain WHY, not just what

Every module starts with a plain-English docstring covering:

1. What this module does (one sentence)
2. Why it exists / what design decision it embodies
3. Where it fits in the pipeline (which stage, what comes before/after)

These docstrings are written to be lifted directly into the paper's
methodology section and presentation slides.

## 5. Deterministic and seeded

Every training/evaluation entry point takes a `seed` from config and sets it
for Python, NumPy, and PyTorch. Reported results state the seed(s) used.

## 6. Separation of concerns across directories

- `src/data/` — turning raw EMBER2024 into model-ready tensors (Stage 1)
- `src/models/` — architectures only; no data loading, no metrics
- `src/evaluation/` — metrics and experiment harness; no model internals
- `src/ablation/` — thin runners that sweep configs; no new logic
- `experiments/` — configs in, results out; no code
- `notebooks/` — exploration only; nothing in `src/` may import a notebook

## 7. Every stage updates the coding guide

When a stage is implemented, its chapter in `docs/CODING_GUIDE.md` is filled
in — what was built, in what order, and why each choice was made — before the
stage is considered done.
