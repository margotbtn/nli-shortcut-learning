# src/analysis/token_attribution.py
"""Token-level attribution for NLI models using Integrated Gradients.

Uses captum's LayerIntegratedGradients to compute per-token importance
scores, then cross-references them with the lexical shortcuts identified
by the lift analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.analysis.shortcut_statistics import LABEL_NAMES


# ============================================================
# 1. Integrated Gradients computation
# ============================================================

def _forward_func(input_ids, attention_mask, model):
    """Forward function for captum: returns logits."""
    return model(input_ids=input_ids, attention_mask=attention_mask).logits


def compute_integrated_gradients(
    model,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[dict],
    device: torch.device,
    label_names: list[str] = LABEL_NAMES,
    n_steps: int = 50,
) -> pd.DataFrame:
    """Compute Integrated Gradients attribution for each token in each example.

    Uses the embedding layer as the attribution target. The baseline is
    the PAD token embedding (standard choice for text IG).

    Args:
        model: Fine-tuned sequence classification model.
        tokenizer: Tokenizer matching the model.
        examples: List of dicts with keys 'hypothesis', 'label' (or 'labels').
        device: Torch device.
        label_names: Ordered label names.
        n_steps: Number of interpolation steps for IG.

    Returns:
        DataFrame with columns: example_idx, token, position,
        attribution, true_label, predicted_label, correct.
    """
    model.eval()

    lig = LayerIntegratedGradients(
        lambda input_ids, attention_mask: _forward_func(input_ids, attention_mask, model),
        model.bert.embeddings.word_embeddings,
    )

    pad_id = tokenizer.pad_token_id
    label_key = "labels" if "labels" in examples[0] else "label"
    records = []

    for idx, example in enumerate(tqdm(examples, desc="Integrated Gradients")):
        hypothesis = example["hypothesis"]
        true_label = example[label_key]

        inputs = tokenizer(hypothesis, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Baseline: all PAD tokens (same shape as input)
        baseline_ids = torch.full_like(input_ids, pad_id)

        # Get model prediction
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted_label = logits.argmax(dim=-1).item()

        # Compute IG w.r.t. the predicted class
        attributions = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=predicted_label,
            n_steps=n_steps,
        )

        # Sum over embedding dimensions → one score per token
        # Shape: (1, seq_len, hidden_dim) → (seq_len,)
        attr_scores = attributions.squeeze(0).sum(dim=-1).cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu())

        # Skip [CLS] and [SEP]
        for pos in range(1, len(tokens) - 1):
            if tokens[pos] == tokenizer.pad_token:
                break
            records.append({
                "example_idx": idx,
                "token": tokens[pos],
                "position": pos,
                "attribution": float(attr_scores[pos]),
                "true_label": label_names[true_label],
                "predicted_label": label_names[predicted_label],
                "correct": predicted_label == true_label,
            })

    return pd.DataFrame(records)


# ============================================================
# 2. Cross-analysis with lift shortcuts
# ============================================================

def aggregate_attributions(
    attr_df: pd.DataFrame,
    lift_lookup: dict[str, dict],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare mean attribution of shortcut vs non-shortcut tokens.

    A token is classified as "shortcut" if it appears in the lift
    lookup (i.e. it passed the min_count and stopword filters in
    the lift analysis).

    Args:
        attr_df: Output of :func:`compute_integrated_gradients`.
        lift_lookup: Output of
            :func:`~src.analysis.shortcut_statistics.build_lift_lookup`.

    Returns:
        per_token: DataFrame with mean attribution per unique token,
            plus its max_lift and is_shortcut flag.
        summary: Dict with keys 'mean_shortcut', 'mean_non_shortcut',
            'ratio', 'n_shortcut', 'n_non_shortcut'.
    """
    df = attr_df.copy()
    df["abs_attribution"] = df["attribution"].abs()

    # Flag shortcut tokens and attach lift
    df["max_lift"] = df["token"].map(
        lambda t: lift_lookup[t]["max_lift"] if t in lift_lookup else 1.0
    )
    df["is_shortcut"] = df["token"].isin(lift_lookup)

    # Per-token aggregation
    per_token = (
        df.groupby("token")
        .agg(
            mean_attr=("abs_attribution", "mean"),
            count=("abs_attribution", "count"),
            max_lift=("max_lift", "first"),
            is_shortcut=("is_shortcut", "first"),
        )
        .reset_index()
        .sort_values("mean_attr", ascending=False)
    )

    # Summary
    shortcut_attr = df[df["is_shortcut"]]["abs_attribution"]
    non_shortcut_attr = df[~df["is_shortcut"]]["abs_attribution"]

    mean_s = shortcut_attr.mean() if len(shortcut_attr) > 0 else 0.0
    mean_ns = non_shortcut_attr.mean() if len(non_shortcut_attr) > 0 else 0.0

    summary = {
        "mean_shortcut": mean_s,
        "mean_non_shortcut": mean_ns,
        "ratio": mean_s / mean_ns if mean_ns > 0 else 0.0,
        "n_shortcut": len(shortcut_attr),
        "n_non_shortcut": len(non_shortcut_attr),
    }

    return per_token, summary


def attribution_by_lift_bin(
    attr_df: pd.DataFrame,
    lift_lookup: dict[str, dict],
    bins: list[float] | None = None,
    bin_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Bin tokens by their max lift and compute mean attribution per bin.

    Args:
        attr_df: Output of :func:`compute_integrated_gradients`.
        lift_lookup: Output of
            :func:`~src.analysis.shortcut_statistics.build_lift_lookup`.
        bins: Bin edges for lift values.
        bin_labels: Labels for each bin.

    Returns:
        DataFrame with columns: lift_bin, n_tokens, mean_attribution.
    """
    if bins is None:
        bins = [0, 1.0, 1.3, 1.6, 2.0, 10]
    if bin_labels is None:
        bin_labels = ["≤1.0 (baseline)", "1.0–1.3", "1.3–1.6", "1.6–2.0", ">2.0 (strong)"]

    df = attr_df.copy()
    df["abs_attribution"] = df["attribution"].abs()
    df["max_lift"] = df["token"].map(
        lambda t: lift_lookup[t]["max_lift"] if t in lift_lookup else 1.0
    )
    df["lift_bin"] = pd.cut(df["max_lift"], bins=bins, labels=bin_labels)

    return (
        df.groupby("lift_bin", observed=True)
        .agg(
            n_tokens=("abs_attribution", "count"),
            mean_attribution=("abs_attribution", "mean"),
        )
        .reset_index()
    )
