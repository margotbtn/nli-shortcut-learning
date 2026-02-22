# Experiment 03 — Shortcut Analysis

## Objective

Identify and quantify the lexical and structural shortcuts in SNLI that enable hypothesis-only prediction, and verify that the trained models actually exploit them.

## Methodology

### 3.1 Lexical shortcuts (lift analysis)

- Tokenized all 549k training hypotheses with BERT WordPiece
- Computed lift = P(label | token) / P(label) for each (token, label) pair
- Applied stopword filtering (124 tokens) and minimum count threshold (≥ 100)
- 2,362 tokens retained after filtering

Lift was chosen over chi-square (biased towards high-frequency tokens) and PMI (biased towards rare tokens) because it directly measures the relative over-representation of a token for a specific label while remaining interpretable.

### 3.2 Structural shortcuts

- Computed hypothesis length (unique tokens), premise–hypothesis overlap ratio, and Jaccard similarity for all 549k training examples
- Fit a logistic regression on 4 structural features to quantify their predictive power

### 3.3 Model behaviour cross-analysis

- Ran inference on the validation set (9,842 examples) with both the pair model (Exp 01) and the hypothesis-only model (Exp 02)
- Cross-referenced predictions with per-example shortcut features (max lift, overlap ratio, hypothesis length)

## Results

### Lexical shortcuts

Top shortcuts per label (lift ≥ 2.0, count ≥ 200):

**Entailment** (14 tokens) — hypernyms and location generalizations:

| Token      | Lift | P(l\|t) | Count  |
|------------|------|---------|--------|
| least      | 3.07 | 92.5%   | 359    |
| outdoors   | 2.61 | 78.8%   | 5,076  |
| instrument | 2.47 | 74.4%   | 949    |
| animal     | 2.27 | 68.4%   | 1,363  |
| outside    | 2.17 | 65.6%   | 14,647 |

**Neutral** (54 tokens) — speculative and elaborative content:

| Token        | Lift | P(l\|t) | Count |
|--------------|------|---------|-------|
| championship | 2.59 | 95.0%   | 342   |
| vacation     | 2.46 | 90.1%   | 598   |
| winning      | 2.45 | 89.8%   | 634   |
| tall         | 2.39 | 87.6%   | 1,836 |
| friends      | 2.02 | 73.8%   | 2,982 |

**Contradiction** (42 tokens) — negation and scene-switching:

| Token    | Lift | P(l\|t) | Count |
|----------|------|---------|-------|
| nobody   | 3.00 | 99.5%   | 2,366 |
| sleeping | 2.59 | 86.2%   | 5,933 |
| no       | 2.54 | 84.5%   | 2,795 |
| alone    | 2.48 | 82.6%   | 1,925 |
| cat      | 2.47 | 82.1%   | 2,444 |

### Structural shortcuts

| Feature        | Entailment | Neutral | Contradiction |
|----------------|------------|---------|---------------|
| hyp_len (mean) | 7.3        | 8.9     | 8.1           |
| overlap_ratio  | 0.618      | 0.472   | 0.438         |
| jaccard        | 0.312      | 0.259   | 0.220         |

Entailment hypotheses are shorter (concise reformulations) with the highest overlap (premise reuse). Neutral hypotheses are the longest (additional narrative). Contradiction hypotheses have the lowest overlap (different vocabulary).

**Logistic regression on structural features alone: 49.0% accuracy** (5-fold CV, vs 33.3% chance).

Strongest standardized coefficients:

| Feature       | Entailment | Neutral | Contradiction |
|---------------|------------|---------|---------------|
| overlap_ratio | +0.478     | −0.198  | −0.280        |
| hyp_len       | −0.471     | +0.364  | +0.107        |

### Model behaviour

**Accuracy by shortcut strength (max lift in hypothesis):**

| Lift bin             | n     | Hyp-only | Pair   |
|----------------------|-------|----------|--------|
| ≤ 1.2 (no shortcut) | 821   | 66.8%    | 89.5%  |
| 1.2 – 1.5           | 2,921 | 66.0%    | 91.0%  |
| 1.5 – 2.0           | 3,600 | 68.7%    | 89.7%  |
| 2.0 – 2.5           | 2,032 | 79.7%    | 91.9%  |
| > 2.5 (strong)       | 468   | 89.1%    | 95.7%  |

The hypothesis-only model's accuracy rises from 66.8% to 89.1% as shortcut strength increases, confirming it exploits the lexical shortcuts identified above.

**Accuracy by overlap ratio:**

| Overlap   | n     | Pair   | Hyp-only |
|-----------|-------|--------|----------|
| 0 – 0.2  | 744   | 89.0%  | 69.9%    |
| 0.2 – 0.4| 2,577 | 90.2%  | 75.7%    |
| 0.4 – 0.6| 3,298 | 89.8%  | 70.6%    |
| 0.6 – 0.8| 2,070 | 91.8%  | 67.7%    |
| 0.8 – 1.0| 1,022 | 94.8%  | 67.1%    |

The pair model improves with overlap (89.0% → 94.8%), confirming overlap serves as a structural shortcut. The hypothesis-only model shows no benefit (as expected: overlap is invisible without the premise).

**Model agreement decomposition:**

| Category              | Count | Pct   |
|-----------------------|-------|-------|
| Both correct          | 6,731 | 68.4% |
| Only pair correct     | 2,205 | 22.4% |
| Only hyp-only correct | 255   | 2.6%  |
| Both wrong            | 651   | 6.6%  |

Feature profiles by group:

| Group             | Mean lift | Mean overlap | Mean hyp_len |
|-------------------|-----------|--------------|--------------|
| Both correct      | 1.735     | 0.505        | 8.1          |
| Only pair correct | 1.589     | 0.547        | 8.3          |
| Both wrong        | 1.631     | 0.471        | 8.6          |

The "both correct" group has higher mean lift (1.735 vs 1.589), confirming that shortcuts drive shared success. The 22.4% "only pair correct" examples represent cases where the premise was genuinely needed.

## Synthesis

### Performance decomposition

| Source                 | Accuracy | Delta     |
|------------------------|----------|-----------|
| Chance                 | 33.3%    | —         |
| + Structural shortcuts | 49.0%    | +15.7 pp  |
| + All hypothesis cues  | 71.0%    | +22.0 pp  |
| + Premise reasoning    | 90.8%    | +19.8 pp  |

Approximately **66%** of above-chance performance is replicable without the premise.

### Shortcut taxonomy

| Label         | Lexical pattern                        | Structural pattern               |
|---------------|----------------------------------------|----------------------------------|
| Entailment    | Hypernyms, location generalizations    | Short hypothesis, high overlap   |
| Neutral       | Speculative/narrative vocabulary       | Long hypothesis, medium overlap  |
| Contradiction | Negation markers, scene-switching      | Low overlap, different vocabulary|

## Implications for next project phase

1. Anti-shortcut evaluation splits should neutralize the identified lexical shortcuts (negation markers, scene-switching tokens, hypernyms, speculative vocabulary)
2. Overlap-based shortcuts can be tested by controlling premise–hypothesis lexical overlap in evaluation splits
3. The 22.4% "only pair correct" examples represent genuine reasoning — Phase 4 should test whether these survive distribution shift
