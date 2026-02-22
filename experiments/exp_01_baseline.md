# Experiment 01 — Baseline NLI Model (Pair)

## Objective

Fine-tune BERT-base-uncased on SNLI with standard premise + hypothesis input to establish baseline performance.

## Setup

| Parameter       | Value                         |
|-----------------|-------------------------------|
| Model           | bert-base-uncased             |
| Dataset         | SNLI (549,367 train examples) |
| Input           | Premise + Hypothesis (pair)   |
| Labels          | entailment, neutral, contradiction |
| Max length      | 128 tokens                    |
| Batch size      | 16 (train) / 32 (eval)       |
| Epochs          | 3                             |
| Optimizer       | AdamW                         |
| Learning rate   | 2e-5                          |
| Warmup steps    | 9,000                         |
| Seed            | 42                            |

## Results

| Metric           | Value  |
|------------------|--------|
| Accuracy         | 90.34% |
| F1 (macro)       | 90.32% |
| Precision (macro)| 90.31% |
| Recall (macro)   | 90.33% |

Evaluated on the SNLI validation split (9,842 examples).

## Discussion

The pair model achieves 90.34% accuracy on the standard SNLI validation set, consistent with published BERT-base results on this benchmark (~90–91%).

This establishes the performance ceiling for our comparison. The key question for subsequent experiments: how much of this accuracy reflects genuine premise–hypothesis reasoning vs. exploitation of dataset artifacts?
