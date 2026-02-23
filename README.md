# Shortcut Learning in Natural Language Inference

## Overview
This repository presents a research-oriented analysis of **shortcut learning** in **Natural Language Inference (NLI)**, focusing on the widely used **SNLI** benchmark.

We show that strong NLI performance can arise from **hypothesis-only biases** — without relying on genuine semantic inference between premise and hypothesis — and that these shortcuts are both statistically quantifiable and mechanistically verifiable through gradient-based attribution.

The goal of this project is not to maximise benchmark accuracy, but to **diagnose failure modes in evaluation benchmarks** and highlight implications for **robustness and AI alignment**.

---

## Motivation
Natural Language Inference is often treated as a proxy for language understanding and reasoning. However, prior work suggests that NLI datasets contain strong annotation artifacts that models can exploit.

From an AI safety perspective, this raises a key concern:

> *High benchmark performance may reflect shortcut exploitation rather than goal-aligned reasoning.*

This project investigates this empirically using controlled input ablations, statistical association measures, and token-level attribution analysis.

---

## Key Research Questions
- To what extent can NLI labels be predicted using the **hypothesis alone**?
- What lexical or structural shortcuts do models rely on?
- Does the model's internal attribution align with the statistical shortcuts identified at the corpus level?
- What does this imply for **robustness** and **goal misgeneralization**?

---

## Main Results

### Phase 1 — Pair model baseline (Exp 01)

| Metric            | Value  |
|-------------------|--------|
| Accuracy          | 90.34% |
| F1 macro          | 90.32% |
| Precision macro   | 90.31% |
| Recall macro      | 90.33% |

Evaluated on the SNLI validation split (9,842 examples). Consistent with published BERT-base results (~90–91%).

### Phase 2 — Hypothesis-only model (Exp 02)

| Condition         | Accuracy | Delta vs chance |
|-------------------|----------|-----------------|
| Chance            | 33.3%    | —               |
| Hypothesis-only   | 70.91%   | +37.6 pp        |
| Pair model        | 90.34%   | +57.0 pp        |

**~66% of above-chance performance is replicable without the premise**: `(70.9 − 33.3) / (90.3 − 33.3) ≈ 0.66`.

### Phase 3 — Shortcut analysis (Exp 03)

**Lexical shortcuts (lift analysis).** 2,362 tokens retained after stopword filtering and a minimum count threshold of 100. Selected high-lift tokens:

| Label         | Token        | Lift | P(label \| token) |
|---------------|--------------|------|-------------------|
| Entailment    | least        | 3.07 | 92.5%             |
| Entailment    | outdoors     | 2.61 | 78.8%             |
| Neutral       | championship | 2.59 | 95.0%             |
| Neutral       | vacation     | 2.46 | 90.1%             |
| Contradiction | nobody       | 3.00 | 99.5%             |
| Contradiction | no           | 2.54 | 84.5%             |

**Structural shortcuts.** Four surface features (hypothesis length, overlap ratio, overlap count, Jaccard) trained in a logistic regression achieve **49.0% 5-fold CV accuracy** (+15.7 pp above chance), without any semantic understanding.

**Model behaviour vs. shortcut strength.** Hypothesis-only accuracy rises monotonically with the maximum token lift in the hypothesis, from 66.8% (no shortcut) to 89.1% (lift > 2.5). The pair model is comparatively stable, showing it does not depend on the same cues.

**Token attribution (Integrated Gradients).** IG attributions computed for 500 validation examples (4,152 token records) confirm that the model has internalized the annotation artifacts:

| Evidence                         | Result                                              |
|----------------------------------|-----------------------------------------------------|
| Shortcut vs. non-shortcut tokens | Mean \|attr\|: 0.398 vs 0.288 — **1.38× higher**  |
| Attribution by lift bin          | 0.288 (lift ≤ 1.0) → 0.800 (lift > 2.0) — **2.8× gradient** |
| Spearman correlation             | ρ = 0.281, p = 4.95e-05 (n = 203 tokens)           |

### Phase 4 — Crossed evaluation on anti-shortcut splits (Exp 05)

Both models evaluated on three splits that neutralize lexical shortcuts:

| Split            | n      | Pair    | Hyp-only | Pair drop | Hyp-only drop |
|------------------|--------|---------|----------|-----------|---------------|
| SNLI validation  | 9,842  | 90.34%  | 70.91%   | —         | —             |
| HANS (OOD)       | 30,000 | 59.39%  | 51.62%   | −30.95 pp | −19.29 pp     |
| SNLI Filtered    | 795    | 89.56%  | 48.18%   | −0.78 pp  | −22.73 pp     |
| SNLI Paraphrased | 1,235  | 68.91%  | 44.37%   | −21.43 pp | −26.54 pp     |

**SNLI Filtered** provides the cleanest isolation: the pair model drops only −0.78 pp (genuine premise reasoning intact), while the hypothesis-only model drops −22.73 pp — a **29× larger drop** that directly quantifies the causal contribution of lexical shortcuts.

**HANS** reveals a distinct structural failure for the pair model: an entailment bias (99.1% entailment recall, 19.68% non-entailment recall) induced by high-overlap entailment training examples.

**SNLI Paraphrased** shows that the pair model also exploits lexical shortcuts opportunistically when present (−21.43 pp on this originally high-lift subset vs −0.78 pp on the low-lift filtered subset).

---

## Project Structure

```text
├── README.md
├── requirements.txt
│
├── experiments/
│   ├── exp_01_baseline.md          # Pair model training run and results
│   ├── exp_02_hypothesis_only.md   # Hypothesis-only ablation results
│   ├── exp_03_shortcut_analysis.md # Lexical and structural shortcut analysis
│   ├── exp_04_attribution.md       # Integrated Gradients cross-analysis
│   └── exp_05_crossed_evaluation.md# Anti-shortcut split evaluation (HANS, filtered, paraphrased)
│
├── notebooks/
│   ├── shortcut_statistics.ipynb   # Lexical lift, structural features, model behaviour
│   └── attribution.ipynb           # Integrated Gradients cross-analysis
│
└── src/
    ├── config.yaml                 # Shared training/evaluation configuration
    ├── configs/
    │   └── base.yaml
    ├── analysis/
    │   ├── shortcut_statistics.py  # Lift computation, structural features, prediction tables
    │   └── token_attribution.py    # Integrated Gradients and lift cross-analysis
    ├── data/
    │   ├── standard.py             # Dataset loading and preprocessing
    │   ├── dataloaders.py          # Tokenisation and DataLoader creation
    │   ├── filtered.py             # Anti-shortcut filtered split (max lift threshold)
    │   ├── paraphrased.py          # Anti-shortcut paraphrased split (WordNet + GPT-2)
    │   └── ood.py                  # HANS OOD evaluation utilities
    ├── models/
    │   ├── load.py                 # Model loading (pretrained / checkpoint)
    │   └── checkpoints.py          # Save/load training checkpoints
    ├── training/
    │   ├── train.py                # Training loop and CLI entry point
    │   ├── eval.py                 # Evaluation loop and CLI entry point
    │   └── metrics.py              # Accuracy, F1, precision, recall, confusion matrix
    └── utils/
        ├── config.py               # YAML config loading with key-path overrides
        ├── logging.py              # Timestamped run directories and loggers
        ├── seed.py                 # Reproducibility seeding (Python / NumPy / PyTorch)
        └── typing.py               # Shared type aliases (Checkpoint, InputView, EvalSet)
```

---

## Methodology

### Training
Both models are fine-tuned from `bert-base-uncased` on the SNLI training set (549,367 examples) under identical hyperparameters:

| Parameter      | Value  |
|----------------|--------|
| Batch size     | 16     |
| Epochs         | 3      |
| Optimizer      | AdamW  |
| Learning rate  | 2e-5   |
| Warmup steps   | 9,000  |
| Max length     | 128    |
| Seed           | 42     |

The only difference between conditions is the input: `pair` (premise + hypothesis) vs. `hypothesis_only` (hypothesis alone).

### Analysis
1. **Lift analysis** — for each (token, label) pair, compute `lift = P(label | token) / P(label)` over all training hypotheses. Tokens are filtered by minimum count (≥ 100) and a stopword list (124 tokens).
2. **Structural features** — hypothesis length, premise–hypothesis overlap ratio, overlap count, and Jaccard similarity, followed by logistic regression cross-validation.
3. **Model behaviour cross-analysis** — both models are run on the full validation set; per-example predictions are joined with shortcut features to measure accuracy as a function of shortcut strength.
4. **Integrated Gradients** — token-level importance scores are computed for a 500-example sample of the validation set (50 interpolation steps, PAD-token baseline) and cross-referenced with the lift lookup.

### Running the code

**Train:**
```bash
python -m src.training.train --input_view pair
python -m src.training.train --input_view hypothesis_only
```

**Evaluate:**
```bash
python -m src.training.eval --input_view pair --checkpoint best --training_run_dir results/<run>
```

**Notebooks** (analysis only — caches are loaded automatically when present):
```bash
jupyter notebook notebooks/shortcut_statistics.ipynb
jupyter notebook notebooks/attribution.ipynb
```

---

## Safety & Alignment Relevance

This work illustrates a concrete instance of **goal misgeneralization**:
- The *intended* objective is semantic inference between sentences.
- The *learned* objective is shortcut exploitation that generalises poorly under distribution shift.

Such failures are directly relevant to:
- evaluation reliability — high benchmark scores do not imply robust reasoning,
- robustness under distribution shift — shortcuts break when the annotation artifact is neutralized,
- misleading signals of alignment or reasoning capability.

The IG analysis (Exp 04) adds a mechanistic dimension: the model does not merely benefit from shortcuts at the distributional level — it has internalized them as its primary decision rule, allocating up to 2.8× more attribution to shortcut tokens than to semantically neutral ones.

Phase 4 (Exp 05) adds a **causal dimension**: the SNLI Filtered split — where the pair model holds at 89.56% while the hypothesis-only model drops to 48.18% — provides direct experimental evidence that shortcut exploitation accounts for most of the hypothesis-only model's above-chance performance. The HANS entailment bias (99.1% entailment recall, 19.68% non-entailment recall for the pair model) illustrates that even the stronger model has internalized a structural heuristic that breaks under OOD conditions.

## Possible Next Steps
- Investigate **debiasing strategies** (product-of-experts, confidence regularization, data augmentation) to reduce shortcut reliance without sacrificing standard accuracy.
- Extend the paraphrased split to cover negation tokens using semantics-preserving transformations (e.g., negation paraphrase with label preservation checks).
- Probe whether debiased models show reduced IG attribution to shortcut tokens.
