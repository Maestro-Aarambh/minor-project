# Project Summary: Classifying Metamorphic Windows PE Malware Using Self-Attention Transformer Model

## 1. What This Project Is

This project builds a machine learning system that classifies Windows PE (Portable Executable) files as malicious or benign, with a specific focus on **metamorphic and evasive malware** — samples that deliberately rewrite their own code structure to escape detection by traditional antivirus engines and standard ML classifiers.

The core question being answered: **does a self-attention transformer detect this kind of evasive malware better than existing methods (CNN, LSTM, tree-based models), and if so, why?**

This is not a survey or literature review — it is an empirical study that trains real models, benchmarks them against each other, and draws conclusions from measured results.

## 2. Dataset

**EMBER2024** — a public benchmark dataset (released 2025) containing 3.2M+ labeled files across six formats. This project uses the **Win32/Win64 PE subset**, which includes:
- A standard temporal train/test split (weeks 1–64)
- An "evasive challenge set" of ~6,300 samples that initially evaded detection by every antivirus engine on VirusTotal — this is the key novelty-enabling resource, since almost no published transformer work has benchmarked against it yet.

## 3. Full Technical Pipeline

**Stage 1 — Data Preparation**
- Download and organize EMBER2024
- Extract three parallel input representations from each PE file:
  - Raw byte-sequence n-grams
  - Disassembled opcode sequences (via Capstone)
  - Static engineered features (EMBER's own header/import/section metadata)
- Handle class imbalance with stratified sampling / class-weighted loss

**Stage 2 — Baseline Models** (establishes the performance floor to beat)
- LightGBM / Random Forest on static features
- 1D CNN on byte sequences
- LSTM/BiLSTM on byte sequences

**Stage 3 — Transformer Model** (the core contribution)
- Self-attention transformer encoder adapted for byte/opcode sequences
- Custom tokenization + positional encoding for code structure
- Trained and tuned on GPU

**Stage 4 — Evaluation**
- Standard metrics: accuracy, precision, recall, F1, ROC-AUC
- Evasive-set-specific detection rate (the key novel comparison)
- Efficiency metrics: training time, inference latency, parameter count, memory
- Robustness: performance across the temporal split (concept drift), and against adversarial perturbations (padding injection, benign section insertion)

**Stage 5 — Ablation Studies**
- Vary attention heads, encoder depth, input representation, positional encoding
- Isolate which architectural choices actually drive performance gains

**Stage 6 — Write-up**
- Results tables/visualizations, discussion, conclusions, final report

## 4. Where the Actual Research Is (vs. Engineering)

This distinction matters — most of the pipeline above is **engineering** (building things that are known to work). The **research** is narrower and lives in a few specific places:

### This is engineering (well-established, low ambiguity, just needs to be built correctly):
- Downloading/preprocessing EMBER2024
- Implementing baseline models (LightGBM, CNN, LSTM) — these architectures and their use on malware are already well documented
- Implementing a standard transformer encoder — the architecture itself is not new
- Computing standard metrics (accuracy, F1, ROC-AUC)

### This is the actual research (open questions, no established answer yet):
1. **Does self-attention actually outperform CNN/LSTM specifically on evasive/metamorphic samples** (not just on the standard test set, where most models already do well)? This is the central hypothesis — nobody has confirmed or denied it yet.
2. **Which architectural components are responsible for any improvement** — is it the attention mechanism itself, the positional encoding, the input representation (opcode vs. byte), or something else? This requires the ablation study to actually isolate causality, not just report a final accuracy number.
3. **How does performance hold up under concept drift** (malware evolving over time) and **adversarial evasion** (padding injection, section manipulation)? This is explicitly flagged in recent literature as under-tested for transformer-based malware classifiers — there's no settled answer.
4. **Efficiency vs. accuracy trade-off** — standardized reporting of this trade-off is currently missing from published malware-classification work; documenting it properly is itself a contribution.

In short: **building the transformer and training it is not the research — the research is in Stages 4 and 5** (systematically testing where and why it does or doesn't outperform baselines, especially on evasive samples, under drift, and under adversarial pressure). That's the part that needs careful experimental design, hypothesis framing, and honest reporting of both positive and negative results — the part a literature review can't substitute for.

## 5. The Exact Research Gap

"Does a transformer work on malware" is not a gap — that has already been answered elsewhere. The actual, precise gap is narrower:

**Existing malware classifiers — including transformer-based ones — are almost exclusively evaluated on standard test sets where samples are already fairly detectable.** Almost none have been systematically tested against samples that specifically evaded every antivirus engine at time of discovery (EMBER2024's evasive challenge set). So the field doesn't actually know whether self-attention's supposed advantage — capturing long-range structural patterns in code — holds up on malware that was engineered to be hard to detect in the first place, or whether that advantage only shows up on "easy" malware where any decent model already performs well.

**The gap, stated plainly:** nobody has confirmed whether transformer-based detection advantages generalize to genuinely evasive/metamorphic malware, or whether they collapse under the same conditions that fool everything else.

### Where the Research Actually Happens (concretely)

**1. The core comparative experiment (the heart of the contribution)**
Train all four models (LightGBM, CNN, LSTM, transformer) and evaluate each **twice** — once on the standard test set, once on the evasive challenge subset — then compare the *performance drop-off* between the two, not just raw accuracy. If the transformer's accuracy drops less than the baselines' when moving from standard to evasive samples, that is evidence of a real advantage. If it drops the same or more, that is also a valid, publishable negative finding — since this specific comparison hasn't been tested by anyone yet.

**2. The ablation study (answers "why")**
Once it's established *whether* the transformer helps, the ablation study answers *why*. Systematically strip away or vary one component at a time — attention heads, layer depth, byte vs. opcode input — and re-run the evasive-set evaluation each time. This identifies which specific architectural piece is doing the work, rather than attributing improvement to "it's a transformer" in general. This is genuinely novel because it hasn't been documented for this dataset/task combination.

**3. Robustness under drift and adversarial pressure (secondary but still open)**
Testing performance across EMBER2024's temporal split (older vs. newer malware) and under simple adversarial perturbations (padding injection, benign section insertion) is explicitly flagged in recent literature as under-tested for transformer-based malware classifiers. Even a modest, honestly-reported set of results here adds new information to a gap nobody has closed.

### What This Means Practically

The literature review and baseline implementation are *setup* — necessary, but not the contribution itself. The actual research happens in three moments:
1. Comparing standard-set vs. evasive-set performance drop-off across all models
2. The ablation study revealing which architectural component matters
3. Reporting robustness under drift/adversarial conditions — even a negative result here is new information

If the final report can state: **"we show that [X] does/doesn't generalize to evasive malware, and it's specifically because of [Y] component"** — that sentence is the research contribution. Everything before it is scaffolding built to reach that conclusion.

## 6. Suggested Build Order
1. Data pipeline (Stage 1)
2. Baselines (Stage 2) — gives you an early, working comparison point
3. Transformer implementation (Stage 3)
4. Full evaluation suite (Stage 4)
5. Ablation studies (Stage 5) — this is where the real findings emerge
6. Write-up (Stage 6)

## 7. Tech Stack
Python, PyTorch, scikit-learn, LightGBM, pefile, Capstone, Weights & Biases/TensorBoard for experiment tracking, GPU (CUDA) for training.
