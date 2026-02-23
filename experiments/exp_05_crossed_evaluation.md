# Experiment 05 — Crossed Evaluation on Anti-Shortcut Splits

## Objective

Measure the accuracy cost of removing lexical shortcuts by evaluating both models (pair and hypothesis-only) on three anti-shortcut splits, and quantify how much of each model's standard performance relies on annotation artifacts.

## Setup

| Split            | n      | Labels                             | Construction                                  |
|------------------|--------|------------------------------------|-----------------------------------------------|
| SNLI validation  | 9,842  | entailment, neutral, contradiction | Reference baseline                            |
| HANS             | 30,000 | entailment, non-entailment         | OOD — heuristic probing                      |
| SNLI Filtered    | 795    | entailment, neutral, contradiction | SNLI val restricted to max token lift ≤ 1.2  |
| SNLI Paraphrased | 1,235  | entailment, neutral, contradiction | High-lift tokens replaced by WordNet synonyms |

**SNLI Filtered** (`src/data/filtered.py`): retains only SNLI validation examples where every hypothesis token has lift < 1.2, removing all strong lexical shortcuts. This produces a subset where hypothesis-only prediction loses its primary signal.

**SNLI Paraphrased** (`src/data/paraphrased.py`): for each example with at least one token of lift > 2.0, replaces those tokens with the first WordNet synonym below lift 1.5. Accepted only if the GPT-2 perplexity ratio (new / original) is < 1.5. Negation and logical tokens (*no*, *nobody*, *never*, etc.) are intentionally skipped — substituting them would change the NLI label. Only examples where at least one substitution was made are included.

**HANS** (`src/data/ood.py`): the Heuristic Analysis for NLI Systems dataset, designed to probe three inference heuristics (lexical overlap, subsequence, constituent). All examples are balanced binary (15,000 entailment, 15,000 non-entailment). SNLI 3-class outputs are collapsed to binary by treating *neutral* and *contradiction* as *non-entailment*.

Both models evaluated with their best checkpoints under identical evaluation code (`src/training/eval.py`).

## Results

### 5.1 Summary — accuracy and drop vs. baseline

| Split            | Pair    | Pair drop    | Hyp-only | Hyp-only drop | Gap (Pair − Hyp) |
|------------------|---------|--------------|----------|---------------|------------------|
| SNLI validation  | 90.34%  | —            | 70.91%   | —             | +19.43 pp        |
| HANS             | 59.39%  | −30.95 pp    | 51.62%   | −19.29 pp     | +7.77 pp         |
| SNLI Filtered    | 89.56%  | −0.78 pp     | 48.18%   | −22.73 pp     | +41.38 pp        |
| SNLI Paraphrased | 68.91%  | −21.43 pp    | 44.37%   | −26.54 pp     | +24.54 pp        |

### 5.2 HANS (out-of-distribution)

30,000 examples, binary (entailment / non-entailment), balanced.

| Metric            | Pair    | Hyp-only |
|-------------------|---------|----------|
| Accuracy          | 59.39%  | 51.62%   |
| F1 (macro)        | 51.79%  | 50.71%   |
| Precision (macro) | 75.43%  | 51.74%   |
| Recall (macro)    | 59.39%  | 51.62%   |

**Per-class recall (confusion matrix):**

| Class          | n      | Pair   | Hyp-only |
|----------------|--------|--------|----------|
| Entailment     | 15,000 | 99.10% | 65.15%   |
| Non-entailment | 15,000 | 19.68% | 38.09%   |

The pair model achieves near-perfect entailment recall (99.1%) but collapses on non-entailment (19.68%), predicting "entailment" for 80% of non-entailment examples. This is the SNLI entailment bias: BERT models learn to predict *entailment* when premise–hypothesis lexical overlap is high — precisely the heuristic HANS exploits in its non-entailment examples. The hypothesis-only model is more balanced but barely above chance (51.62%), consistent with HANS hypotheses carrying minimal standalone signal.

### 5.3 SNLI Filtered (lexically low-shortcut subset)

795 examples (all hypothesis tokens: max lift < 1.2), 3-class.

| Metric            | Pair    | Hyp-only |
|-------------------|---------|----------|
| Accuracy          | 89.56%  | 48.18%   |
| F1 (macro)        | 88.73%  | 35.07%   |
| Precision (macro) | 89.05%  | 44.67%   |
| Recall (macro)    | 88.46%  | 39.37%   |

**Per-class recall (confusion matrix):**

| Class         | n   | Pair   | Hyp-only |
|---------------|-----|--------|----------|
| Entailment    | 356 | 94.94% | 90.73%   |
| Neutral       | 222 | 82.88% | 11.26%   |
| Contradiction | 217 | 87.56% | 16.13%   |

The pair model is **virtually unchanged** by lexical filtering (−0.78 pp), confirming it can perform genuine premise–hypothesis reasoning independently of hypothesis-side lexical shortcuts. The hypothesis-only model drops −22.73 pp, revealing that the bulk of its above-chance performance was driven by lexical cues. Without shortcut tokens, its neutral and contradiction recall collapses to near chance (11–16%), while entailment recall remains high (90.73%) — likely reflecting structural properties of entailment hypotheses (shorter, more general phrasings) that survive lexical filtering.

### 5.4 SNLI Paraphrased (synonym-substituted)

1,235 examples (high-lift tokens replaced, GPT-2 perplexity-filtered), 3-class.

| Metric            | Pair    | Hyp-only |
|-------------------|---------|----------|
| Accuracy          | 68.91%  | 44.37%   |
| F1 (macro)        | 68.83%  | 35.84%   |
| Precision (macro) | 69.88%  | 46.63%   |
| Recall (macro)    | 69.86%  | 40.47%   |

**Per-class recall (confusion matrix):**

| Class         | n   | Pair   | Hyp-only |
|---------------|-----|--------|----------|
| Entailment    | 479 | 57.62% | 87.89%   |
| Neutral       | 373 | 70.51% | 14.21%   |
| Contradiction | 383 | 81.46% | 19.32%   |

Both models degrade significantly. The pair model drops −21.43 pp, larger than on SNLI Filtered, because the paraphrased split is composed exclusively of examples that originally contained high-lift tokens (lift > 2.0) — precisely the examples where both models benefited most from lexical shortcuts. After synonym substitution, the pair model loses those cues.

The hypothesis-only model shows an inverted per-class pattern: high entailment recall (87.89%) but near-chance neutral and contradiction recall (14–19%). This reflects the construction of the split: negation tokens (*nobody*, *no*) are excluded from substitution, so contradiction examples in this split had their *other* shortcuts replaced (*sleeping*, *cat*, *alone* → synonyms). The model exploited those non-negation shortcuts and now fails without them. The high entailment recall persists because entailment hypotheses often retain subtle structural predictability even after synonym substitution.

## Synthesis

### Performance decomposition across splits

The SNLI Filtered split provides the cleanest isolation of lexical shortcut contribution. The **29× larger accuracy drop** for the hypothesis-only model (−22.73 pp vs −0.78 pp for the pair model) directly measures the causal contribution of lexical shortcuts:

- Pair model without shortcuts: **89.56%** — genuine premise reasoning capacity largely intact.
- Hypothesis-only model without shortcuts: **48.18%** — near chance, confirming that lexical shortcuts drove most of its above-chance performance.

### Anti-shortcut split comparison

| Split            | What is neutralized                       | Residual signal for hyp-only     |
|------------------|-------------------------------------------|----------------------------------|
| SNLI Filtered    | Tokens with lift > 1.2                   | Structural (length, generality)  |
| SNLI Paraphrased | Non-negation tokens with lift > 2.0      | Negation tokens, structural cues |
| HANS             | In-distribution heuristics (all 3 types) | Near-zero OOD signal             |

### Implications

1. The pair model's robustness on SNLI Filtered (−0.78 pp) confirms it has a genuine reasoning capacity that does not depend on hypothesis-side lexical cues — but this capacity is masked under standard evaluation by the large proportion of shortcut-exploitable examples.
2. The pair model's drop on SNLI Paraphrased (−21.43 pp) reveals a softer dependence: it also exploits lexical shortcuts opportunistically when they are present, consistent with the Exp 03 finding that pair model accuracy increases with shortcut strength.
3. The HANS entailment bias identifies a separate structural failure mode — an over-learned heuristic from SNLI's overlap-heavy entailment examples — that lexical filtering does not address.
4. The hypothesis-only model's high entailment recall under filtering and paraphrasing suggests that entailment-associated hypothesis structure (conciseness, generality) is partially learned as a secondary, structural shortcut that survives lexical neutralization.
