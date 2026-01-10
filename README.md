# Shortcut Learning in Natural Language Inference

## Overview
This repository presents a research-oriented analysis of **shortcut learning** in **Natural Language Inference (NLI)**, focusing on the widely used **SNLI/MNLI** benchmarks.

We show that strong NLI performance can arise from **hypothesis-only biases**, without relying on genuine semantic inference between premise and hypothesis, and that these shortcuts lead to severe failures under controlled **distribution shifts**.

The goal of this project is not to maximize benchmark accuracy, but to **diagnose failure modes in evaluation benchmarks** and highlight implications for **robustness and AI alignment**.

---

## Motivation
Natural Language Inference is often treated as a proxy for language understanding and reasoning. However, prior work suggests that NLI datasets contain strong annotation artifacts that models can exploit.

From an AI safety perspective, this raises a key concern:

> *High benchmark performance may reflect shortcut exploitation rather than goal-aligned reasoning.*

This project investigates this issue empirically using controlled input ablations and dataset splits.

---

## Key Research Questions
- To what extent can NLI labels be predicted using the **hypothesis alone**?
- What lexical or structural shortcuts do models rely on?
- How do these shortcuts fail under **distribution shift**?
- What does this imply for **robustness** and **goal misgeneralization**?

---

## Experimental Setup

### Dataset
- **SNLI** (primary benchmark)
- Optional extension to **MNLI**

We construct multiple evaluation splits:
- **Standard**: original validation set
- **Hypothesis-only**: premise removed or masked
- **Anti-shortcut / OOD**: spurious lexical correlations neutralized or inverted

Details on dataset construction are provided in `data/README.md`.

---

### Models
- **BERT-base-uncased**, fine-tuned for 3-way NLI classification
- Same architecture, hyperparameters, and training setup across all conditions to ensure controlled comparisons

---

## Methodology
We follow a diagnostic evaluation protocol:

1. **Fine-tune BERT on the standard NLI training set** (premise + hypothesis).
2. **Evaluate the same trained model** on:
   - the standard validation set,
   - a hypothesis-only version of the validation set,
   - anti-shortcut / out-of-distribution (OOD) splits.
3. (Optional baseline) Train a **hypothesis-only model** to verify that label information is present in the hypothesis alone.

The model is **not retrained** on OOD or anti-shortcut splits, allowing us to directly measure robustness under distribution shift.

---

## Main Results (Summary)
- Hypothesis-only evaluations achieve accuracy far above chance, indicating strong dataset biases.
- Performance collapses on anti-shortcut and OOD splits.
- Analysis reveals reliance on simple lexical cues (e.g. negation markers) rather than semantic inference.

These results demonstrate that standard NLI accuracy is **not a reliable indicator of robust reasoning**.

---

## Analysis
The repository includes:
- Statistical analysis of token–label correlations
- Controlled dataset splits targeting specific shortcuts
- Optional representation probing and attention analysis

See the `src/analysis/` directory for details.

---

## Safety & Alignment Relevance
This work illustrates a concrete instance of **goal misgeneralization**:
- The intended objective is semantic inference between sentences.
- The learned objective is shortcut exploitation that generalizes poorly.

Such failures are directly relevant to:
- evaluation reliability,
- robustness under distribution shift,
- misleading signals of alignment or reasoning capability.

---

## Project Structure

```text
nli-shortcut-learning/
│
├── README.md                 # Project overview and results
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── raw/                  # Original SNLI/MNLI data
│   ├── processed/
│   │   ├── standard/         # Standard evaluation split
│   │   ├── hypothesis_only/  # Premise removed
│   │   ├── anti_shortcut/    # Spurious correlations neutralized
│   │   └── ood/              # Distribution-shifted splits
│   └── README.md             # Dataset and split descriptions
│
├── src/
│   ├── configs/              # Training and evaluation configs
│   ├── data/                 # Data loading and split creation
│   ├── models/               # NLI models and hypothesis-only variants
│   ├── training/             # Training and evaluation loops
│   ├── analysis/             # Shortcut and representation analysis
│   └── utils/                # Logging, seeding, helpers
│
├── experiments/
│   ├── exp_01_baseline.md
│   ├── exp_02_hypothesis_only.md
│   ├── exp_03_ood_shift.md
│   └── exp_04_mitigation.md
│
├── scripts/
│   ├── run_baseline.sh
│   ├── run_hypothesis_only.sh
│   └── run_ood.sh
│
├── results/
│   ├── tables/               # Quantitative results
│   ├── figures/              # Plots and visualizations
│   └── logs/                 # Training logs
│
└── notebooks/                # Exploratory analysis and visualization
```
