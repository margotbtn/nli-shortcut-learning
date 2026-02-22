# src/training/eval.py
"""Evaluation loop and CLI entry point for assessing sequence classification models on NLI datasets."""
from __future__ import annotations

from typing import Any, Callable
import argparse
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.load import load_model
from src.training.metrics import compute_metrics
from src.data.dataloaders import prepare_dataloader
from src.data.filtered import prepare_filtered_dataloader
from src.data.paraphrased import prepare_paraphrased_dataloader
from src.data.ood import hans_pred_transform, prepare_hans_dataloader
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger, make_run_dir
from src.utils.seed import set_seed
from src.utils.typing import InputView, Checkpoint, EvalSet


@torch.no_grad()
def evaluate(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    num_labels: int,
    logger: logging.Logger,
    pred_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[dict[str, float], np.ndarray, float]:
    """Runs evaluation loop and computes metrics.

    Args:
        model: The sequence classification model to evaluate.
        dataloader: DataLoader for the evaluation dataset.
        device: Device to run the evaluation on.
        num_labels: Number of unique labels used for metric computation.
            For HANS this is 2 (binary), even though the model outputs 3 logits.
        logger: Logger for logging evaluation progress.
        pred_transform: Optional function applied to raw argmax predictions before
            metric computation (e.g. `hans_pred_transform` to collapse a 3-class
            SNLI model output to HANS binary labels).

    Returns:
        Computed metrics (accuracy, recall, precision, f1).
        Confusion matrix.
        Average loss over the evaluated dataset.
    """
    model.eval()

    all_preds = []
    all_labels = []
    num_steps = len(dataloader)
    epoch_loss = 0.0

    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        loss = outputs.loss
        epoch_loss += loss.item()

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    if pred_transform is not None:
        y_pred = pred_transform(y_pred)

    metrics, cm = compute_metrics(y_true, y_pred, num_labels)

    return metrics, cm, epoch_loss / num_steps


def save_results(
    run_dir: str | Path,
    config: dict[str, Any],
    labels: list[str],
    metrics: dict[str, float],
    confusion_matrix: np.ndarray,
) -> None:
    """Saves config and evaluation results.

    Args:
        run_dir: Directory to save the results in.
        config: Configuration dictionary used for evaluation.
        labels: List of label names.
        metrics: Dictionary of computed metrics.
        confusion_matrix: Confusion matrix as a numpy array.
    """
    run_dir = Path(run_dir)

    enriched_config = {**config, "labels": labels}
    with open(run_dir / "config_with_labels.json", "w", encoding="utf-8") as f:
        json.dump(enriched_config, f, indent=2, sort_keys=True)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    np.save(run_dir / "confusion_matrix.npy", confusion_matrix)


def main(
    input_view: InputView = "pair",
    checkpoint: Checkpoint = "pretrained",
    training_run_dir: str | None = None,
    mode: EvalSet = "standard",
    split: str | None = None,
) -> None:
    """High-level evaluation entry point.

    Args:
        input_view: "pair" or "hypothesis_only".
        checkpoint: Which model checkpoint to load ("pretrained", "best", "latest").
        training_run_dir: Path to the training run directory (required for best/latest).
        mode: Which evaluation split to use:
            "standard"         — standard SNLI split via prepare_dataloader,
            "snli-filtered"    — SNLI validation filtered to low-lift examples,
            "snli-paraphrased" — SNLI validation with shortcut tokens substituted,
            "hans"             — HANS OOD benchmark (binary evaluation).
        split: Dataset split override (e.g. "validation"). Falls back to config value.
    """
    overrides: dict = {
        ("data", "input_view"): input_view,
        ("eval", "checkpoint"): checkpoint,
        ("eval", "training_run_dir"): training_run_dir,
        ("eval", "mode"): mode,
    }
    if split is not None:
        overrides[("eval", "split")] = split

    cfg = load_yaml_config("src/config.yaml", overrides)

    set_seed(cfg["random"]["seed"])

    if cfg["eval"]["checkpoint"] != "pretrained" and not cfg["eval"]["training_run_dir"]:
        raise ValueError("training_run_dir is required when checkpoint is not 'pretrained'.")

    run_name = (
        f"eval_{cfg['eval']['mode']}_"
        f"{input_view}_"
        f"{cfg['data']['dataset_name'].split('/')[-1]}_"
        f"{cfg['model']['pretrained_model_name'].replace('/', '-')}"
    )
    run_dir = make_run_dir(base_dir=Path("results"), run_name=run_name)

    logger = get_logger(name=run_name, log_file=run_dir / "run.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting evaluation...")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["pretrained_model_name"])

    # -----------------------------------------------------------------------
    # Dataloader selection
    # -----------------------------------------------------------------------
    logger.info("Preparing dataloader (mode=%s)...", cfg["eval"]["mode"])
    eval_mode: EvalSet = cfg["eval"]["mode"]
    transform: Callable[[np.ndarray], np.ndarray] | None = None

    if eval_mode == "standard":
        dataloader, labels = prepare_dataloader(
            dataset_name=cfg["data"]["dataset_name"],
            split=cfg["eval"]["split"],
            input_view=cfg["data"]["input_view"],
            tokenizer=tokenizer,
            batch_size=cfg["eval"]["batch_size"],
            max_length=cfg["data"]["max_length"],
            shuffle=False,
            keep_text=cfg["eval"]["keep_text"],
            num_proc=cfg["data"]["num_proc"],
            num_workers=cfg["data"]["num_workers"],
        )

    elif eval_mode == "hans":
        dataloader, labels = prepare_hans_dataloader(
            split=cfg["eval"]["split"],
            input_view=cfg["data"]["input_view"],
            tokenizer=tokenizer,
            batch_size=cfg["eval"]["batch_size"],
            max_length=cfg["data"]["max_length"],
            keep_text=cfg["eval"]["keep_text"],
            num_proc=cfg["data"]["num_proc"],
            num_workers=cfg["data"]["num_workers"],
        )
        transform = hans_pred_transform

    elif eval_mode == "snli-filtered":
        dataloader, labels = prepare_filtered_dataloader(
            dataset_name=cfg["data"]["dataset_name"],
            split=cfg["eval"]["split"],
            input_view=cfg["data"]["input_view"],
            tokenizer=tokenizer,
            lift_scores_path=cfg["eval"]["lift_scores_path"],
            max_lift_threshold=cfg["eval"]["max_lift_threshold"],
            batch_size=cfg["eval"]["batch_size"],
            max_length=cfg["data"]["max_length"],
            keep_text=cfg["eval"]["keep_text"],
            num_proc=cfg["data"]["num_proc"],
            num_workers=cfg["data"]["num_workers"],
        )

    elif eval_mode == "snli-paraphrased":
        dataloader, labels = prepare_paraphrased_dataloader(
            dataset_name=cfg["data"]["dataset_name"],
            split=cfg["eval"]["split"],
            input_view=cfg["data"]["input_view"],
            tokenizer=tokenizer,
            lift_scores_path=cfg["eval"]["lift_scores_path"],
            max_lift_threshold=cfg["eval"]["max_lift_threshold"],
            perplexity_ratio_threshold=cfg["eval"]["perplexity_ratio_threshold"],
            batch_size=cfg["eval"]["batch_size"],
            max_length=cfg["data"]["max_length"],
            keep_text=cfg["eval"]["keep_text"],
            num_proc=cfg["data"]["num_proc"],
            num_workers=cfg["data"]["num_workers"],
        )

    else:
        raise ValueError(f"Unknown eval mode: {eval_mode!r}")

    num_labels = len(labels)
    logger.info("Dataset size: %d examples, %d labels.", len(dataloader.dataset), num_labels)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    logger.info("Loading model...")
    model = load_model(
        model_name=cfg["model"]["pretrained_model_name"],
        num_labels=num_labels if eval_mode != "hans" else 3,
        device=device,
        checkpoint=cfg["eval"]["checkpoint"],
        run_dir=Path(cfg["eval"]["training_run_dir"]) if cfg["eval"]["training_run_dir"] else None,
    )

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    logger.info("Running evaluation...")
    metrics, cm, _ = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        num_labels=num_labels,
        logger=logger,
        pred_transform=transform,
    )

    logger.info("Saving results...")
    save_results(
        run_dir=run_dir,
        config=cfg,
        labels=labels,
        metrics=metrics,
        confusion_matrix=cm,
    )

    logger.info("Evaluation complete. Metrics: %s", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a sequence classification model.")
    parser.add_argument(
        "--input_view",
        choices=["pair", "hypothesis_only"],
        default="pair",
        help="Train and evaluate the model either on both sentences or only the hypothesis.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "latest", "pretrained"],
        default="pretrained",
        help="Checkpoint to evaluate (default: pretrained).",
    )
    parser.add_argument(
        "--training_run_dir",
        default=None,
        help="Training run directory for latest/best checkpoints.",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "snli-filtered", "snli-paraphrased", "hans"],
        default="standard",
        help="Evaluation split / mode (default: standard).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split override (e.g. 'validation'). Falls back to config value.",
    )
    args = parser.parse_args()
    main(args.input_view, args.checkpoint, args.training_run_dir, args.mode, args.split)
