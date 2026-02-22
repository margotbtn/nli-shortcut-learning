# Experiment 02 — Hypothesis-Only Model

## Objective

Train an identical BERT model using only the hypothesis (premise removed) to quantify the extent of hypothesis-only bias in SNLI.

## Setup

| Parameter       | Value                         |
|-----------------|-------------------------------|
| Model           | bert-base-uncased             |
| Dataset         | SNLI (549,367 train examples) |
| Input           | Hypothesis only               |
| Labels          | entailment, neutral, contradiction |
| All other params| Identical to Experiment 01    |

## Results

| Metric           | Hypothesis-only | Pair (Exp 01) | Chance |
|------------------|-----------------|---------------|--------|
| Accuracy         | 70.91%          | 90.34%        | 33.3%  |
| F1 (macro)       | 70.86%          | 90.32%        | —      |
| Precision (macro)| 70.95%          | 90.31%        | —      |
| Recall (macro)   | 70.86%          | 90.33%        | —      |

Evaluated on the SNLI validation split (9,842 examples).

## Discussion

The hypothesis-only model achieves 70.91% accuracy — more than twice chance level (33.3%) — without ever seeing the premise. This confirms that SNLI hypotheses contain strong annotation artifacts sufficient to predict the label a majority of the time.

### Preliminary gap decomposition

| Source                    | Accuracy | Delta     |
|---------------------------|----------|-----------|
| Chance                    | 33.3%    | —         |
| → Hypothesis-only model   | 70.9%    | +37.6 pp  |
| → Pair model              | 90.3%    | +19.4 pp  |

Shortcuts account for approximately 66% of the pair model's above-chance performance: `(70.9 − 33.3) / (90.3 − 33.3) ≈ 0.66`.

This motivates the following project phase: identifying exactly what these shortcuts are (lexical cues, structural patterns, or both).
