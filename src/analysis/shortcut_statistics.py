# src/analysis/shortcut_statistics.py
"""Shortcut detection statistics for NLI datasets.

Provides functions to quantify lexical and structural shortcuts
in NLI training data and cross-reference them with model predictions.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase


### Constants

LABEL_NAMES: list[str] = ["entailment", "neutral", "contradiction"]

STOPWORDS: set[str] = {
    # punctuation / sub-word artifacts
    ".", ",", "!", "?", "'", '"', "-", "##s", "##ed", "##ing", "##er",
    # articles / determiners
    "a", "an", "the",
    # prepositions
    "in", "on", "at", "by", "of", "to", "for", "with", "from", "into",
    "through", "about", "between", "after", "before", "during", "under",
    "over", "up", "down", "out", "off", "around", "near",
    # conjunctions / function words
    "and", "or", "but", "if", "as", "while", "that", "which", "who",
    "than", "so", "because",
    # pronouns
    "i", "he", "she", "it", "we", "they", "his", "her", "their",
    "its", "our", "my", "your", "him", "them", "us",
    # auxiliary / modal verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "may", "might", "shall", "should", "must",
    # common adverbs / particles
    "not", "also", "just", "very", "more", "most", "all", "some", "any",
    "both", "each", "other", "only", "same", "such", "even", "still",
    "there", "here", "then", "when", "where", "how", "what",
    # common content words with no discriminative value in SNLI
    "man", "woman", "people", "person", "child", "children",
    "one", "two", "three", "many", "few", "s",
}


### Lexical shortcuts (lift)

def compute_token_label_counts(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    label_names: list[str] = LABEL_NAMES,
) -> dict[str, dict[str, int]]:
    """Count unique-token occurrences per label across all hypotheses.

    For each example, tokenizes the hypothesis and increments the count
    of each *unique* token for the example's label.

    Args:
        dataset: HuggingFace Dataset with 'hypothesis' and 'label' columns.
        tokenizer: Tokenizer to use.
        label_names: Ordered label names mapping int labels to strings.

    Returns:
        Nested dict ``{token: {label_name: count}}``.
    """
    special = set(tokenizer.all_special_tokens)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    label_key = "labels" if "labels" in dataset.column_names else "label"

    for example in tqdm(dataset, desc="Counting tokens"):
        label_name = label_names[example[label_key]]
        tokens = set(tokenizer.tokenize(example["hypothesis"].lower())) - special
        for token in tokens:
            counts[token][label_name] += 1

    return {t: dict(lc) for t, lc in counts.items()}


def compute_lift(
    token_label_counts: dict[str, dict[str, int]],
    label_names: list[str] = LABEL_NAMES,
    min_count: int = 100,
    stopwords: set[str] | None = None,
) -> pd.DataFrame:
    """Compute lift scores for each (token, label) pair.

    ``lift(t, l) = P(l | t) / P(l)``

    Args:
        token_label_counts: Output of :func:`compute_token_label_counts`.
        label_names: Ordered label names.
        min_count: Minimum total occurrences to keep a token.
        stopwords: Tokens to exclude. Defaults to :data:`STOPWORDS`.

    Returns:
        DataFrame with columns: token, label, count, total_count,
        p_label_given_token, p_label, lift, concentration.
    """
    if stopwords is None:
        stopwords = STOPWORDS

    label_totals = {
        l: sum(v.get(l, 0) for v in token_label_counts.values())
        for l in label_names
    }
    N = sum(label_totals.values())

    rows = []
    for token, counts in token_label_counts.items():
        total = sum(counts.get(l, 0) for l in label_names)
        if total < min_count or token in stopwords:
            continue
        for l in label_names:
            c = counts.get(l, 0)
            p_label_given_token = c / total
            p_label = label_totals[l] / N
            lift = p_label_given_token / p_label if p_label > 0 else 0.0
            rows.append({
                "token": token,
                "label": l,
                "count": c,
                "total_count": total,
                "p_label_given_token": p_label_given_token,
                "p_label": p_label,
                "lift": lift,
            })

    df = pd.DataFrame(rows)
    conc = df.groupby("token")["p_label_given_token"].max().rename("concentration")
    df = df.join(conc, on="token")
    return df


def top_shortcuts(
    lift_df: pd.DataFrame,
    top_k: int = 20,
    label_names: list[str] = LABEL_NAMES,
) -> dict[str, pd.DataFrame]:
    """Extract the top-K highest-lift tokens per label.

    Args:
        lift_df: Output of :func:`compute_lift`.
        top_k: Number of tokens to return per label.
        label_names: Ordered label names.

    Returns:
        Dict mapping each label to a DataFrame of its top shortcuts,
        sorted by descending lift.
    """
    result = {}
    for label in label_names:
        result[label] = (
            lift_df[lift_df["label"] == label]
            .sort_values("lift", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
    return result


### Structural shortcuts

def compute_structural_features(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    label_names: list[str] = LABEL_NAMES,
) -> pd.DataFrame:
    """Compute structural features for each example.

    Features:
        - hyp_len: number of unique BERT tokens in the hypothesis
        - prem_len: number of unique BERT tokens in the premise
        - overlap_count: ``|premise_tokens & hypothesis_tokens|``
        - overlap_ratio: ``overlap_count / hyp_len``
        - jaccard: ``overlap_count / |premise_tokens | hypothesis_tokens|``
        - label: string label name

    Args:
        dataset: HuggingFace Dataset with premise, hypothesis, label.
        tokenizer: Tokenizer for splitting into tokens.
        label_names: Ordered label names.

    Returns:
        DataFrame with one row per example.
    """
    special = set(tokenizer.all_special_tokens)
    label_key = "labels" if "labels" in dataset.column_names else "label"
    records = []

    for example in tqdm(dataset, desc="Structural features"):
        prem_tokens = set(tokenizer.tokenize(example["premise"].lower())) - special
        hyp_tokens = set(tokenizer.tokenize(example["hypothesis"].lower())) - special

        overlap = prem_tokens & hyp_tokens
        union = prem_tokens | hyp_tokens
        hyp_len = len(hyp_tokens)

        records.append({
            "label": label_names[example[label_key]],
            "hyp_len": hyp_len,
            "prem_len": len(prem_tokens),
            "overlap_count": len(overlap),
            "overlap_ratio": len(overlap) / hyp_len if hyp_len > 0 else 0.0,
            "jaccard": len(overlap) / len(union) if len(union) > 0 else 0.0,
        })

    return pd.DataFrame(records)


def structural_predictive_power(
    structural_df: pd.DataFrame,
    feature_names: list[str] | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[float, float, pd.DataFrame]:
    """Fit a logistic regression on structural features and return CV accuracy.

    Args:
        structural_df: Output of :func:`compute_structural_features`.
        feature_names: Columns to use as features.
        cv_folds: Number of cross-validation folds.
        random_state: Random seed.

    Returns:
        mean_accuracy: Mean CV accuracy.
        std_accuracy: Std of CV accuracy.
        coefficients: DataFrame of shape (n_features, n_labels) with
            standardized logistic regression coefficients.
    """
    if feature_names is None:
        feature_names = ["hyp_len", "overlap_count", "overlap_ratio", "jaccard"]

    X = structural_df[feature_names].values
    le = LabelEncoder()
    y = le.fit_transform(structural_df["label"].values)

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=random_state),
    )
    scores = cross_val_score(pipe, X, y, cv=cv_folds, scoring="accuracy")

    pipe.fit(X, y)
    lr = pipe.named_steps["logisticregression"]
    coef_df = pd.DataFrame(lr.coef_, index=le.classes_, columns=feature_names).T

    return scores.mean(), scores.std(), coef_df


### Model behaviour cross-analysis

def build_lift_lookup(lift_df: pd.DataFrame) -> dict[str, dict]:
    """Build a token → {best_label, max_lift} lookup from lift scores.

    For each token, keeps only the label with the highest lift.

    Args:
        lift_df: Output of :func:`compute_lift`.

    Returns:
        Dict ``{token: {"best_label": str, "max_lift": float}}``.
    """
    idx_max = lift_df.groupby("token")["lift"].idxmax()
    best = lift_df.loc[idx_max, ["token", "label", "lift"]].set_index("token")
    best.columns = ["best_label", "max_lift"]
    return best.to_dict("index")


def compute_example_shortcut_features(
    premise: str,
    hypothesis: str,
    tokenizer: PreTrainedTokenizerBase,
    lift_lookup: dict[str, dict],
) -> dict[str, float | str]:
    """Compute shortcut features for a single example.

    Args:
        premise: Premise text.
        hypothesis: Hypothesis text.
        tokenizer: Tokenizer.
        lift_lookup: Output of :func:`build_lift_lookup`.

    Returns:
        Dict with keys: hyp_len, overlap_ratio, max_lift, max_lift_token.
    """
    special = set(tokenizer.all_special_tokens)
    hyp_tokens = set(tokenizer.tokenize(hypothesis.lower())) - special
    prem_tokens = set(tokenizer.tokenize(premise.lower())) - special

    overlap = prem_tokens & hyp_tokens
    hyp_len = len(hyp_tokens)

    max_lift = 1.0
    max_lift_token = ""
    for t in hyp_tokens:
        info = lift_lookup.get(t)
        if info and info["max_lift"] > max_lift:
            max_lift = info["max_lift"]
            max_lift_token = t

    return {
        "hyp_len": hyp_len,
        "overlap_ratio": len(overlap) / hyp_len if hyp_len > 0 else 0.0,
        "max_lift": max_lift,
        "max_lift_token": max_lift_token,
    }


def build_prediction_table(
    dataset,
    model_pair,
    model_hyp,
    tokenizer: PreTrainedTokenizerBase,
    lift_lookup: dict[str, dict],
    device: torch.device,
    label_names: list[str] = LABEL_NAMES,
) -> pd.DataFrame:
    """Run inference with both models and attach shortcut features.

    Args:
        dataset: Validation split with premise, hypothesis, label.
        model_pair: Pair-input trained model.
        model_hyp: Hypothesis-only trained model.
        tokenizer: Shared tokenizer.
        lift_lookup: Output of :func:`build_lift_lookup`.
        device: Torch device.
        label_names: Ordered label names.

    Returns:
        DataFrame with one row per validation example.
    """
    label_key = "labels" if "labels" in dataset.column_names else "label"
    results = []

    with torch.no_grad():
        for example in tqdm(dataset, desc="Inference"):
            premise = example["premise"]
            hypothesis = example["hypothesis"]
            true_label = example[label_key]

            features = compute_example_shortcut_features(
                premise, hypothesis, tokenizer, lift_lookup,
            )

            # Pair model
            inputs_pair = tokenizer(
                premise, hypothesis,
                return_tensors="pt", truncation=True, max_length=128,
            )
            inputs_pair = {k: v.to(device) for k, v in inputs_pair.items()}
            pred_pair = model_pair(**inputs_pair).logits.argmax(dim=-1).item()

            # Hypothesis-only model
            inputs_hyp = tokenizer(
                hypothesis,
                return_tensors="pt", truncation=True, max_length=128,
            )
            inputs_hyp = {k: v.to(device) for k, v in inputs_hyp.items()}
            pred_hyp = model_hyp(**inputs_hyp).logits.argmax(dim=-1).item()

            results.append({
                "true_label": label_names[true_label],
                "pred_pair": label_names[pred_pair],
                "pred_hyp": label_names[pred_hyp],
                "correct_pair": pred_pair == true_label,
                "correct_hyp": pred_hyp == true_label,
                **features,
            })

    return pd.DataFrame(results)


def accuracy_by_bin(
    pred_df: pd.DataFrame,
    feature: str,
    bins: list[float],
    bin_labels: list[str],
) -> pd.DataFrame:
    """Bin examples by a feature and compute per-bin accuracy.

    Args:
        pred_df: Output of :func:`build_prediction_table`.
        feature: Column name to bin on.
        bins: Bin edges.
        bin_labels: Human-readable labels for each bin.

    Returns:
        DataFrame with columns: bin, n, acc_pair, acc_hyp.
    """
    pred_df = pred_df.copy()
    pred_df["_bin"] = pd.cut(pred_df[feature], bins=bins, labels=bin_labels)
    return (
        pred_df.groupby("_bin", observed=True)
        .agg(n=("correct_pair", "count"),
             acc_pair=("correct_pair", "mean"),
             acc_hyp=("correct_hyp", "mean"))
        .reset_index()
        .rename(columns={"_bin": "bin"})
    )


def agreement_analysis(
    pred_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify examples into agreement categories and compute feature profiles.

    Categories: both_correct, only_pair, only_hyp, both_wrong.

    Args:
        pred_df: Output of :func:`build_prediction_table`.

    Returns:
        summary: DataFrame with columns category, count, pct.
        profiles: DataFrame with columns category, mean_lift,
            mean_overlap, mean_hyp_len.
    """
    df = pred_df.copy()
    conditions = [
        ("both_correct", df["correct_pair"] & df["correct_hyp"]),
        ("only_pair", df["correct_pair"] & ~df["correct_hyp"]),
        ("only_hyp", ~df["correct_pair"] & df["correct_hyp"]),
        ("both_wrong", ~df["correct_pair"] & ~df["correct_hyp"]),
    ]

    summary_rows = []
    profile_rows = []
    for name, mask in conditions:
        sub = df[mask]
        summary_rows.append({
            "category": name,
            "count": len(sub),
            "pct": len(sub) / len(df),
        })
        profile_rows.append({
            "category": name,
            "mean_lift": sub["max_lift"].mean() if len(sub) > 0 else 0.0,
            "mean_overlap": sub["overlap_ratio"].mean() if len(sub) > 0 else 0.0,
            "mean_hyp_len": sub["hyp_len"].mean() if len(sub) > 0 else 0.0,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(profile_rows)
