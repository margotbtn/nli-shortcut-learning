from __future__ import annotations

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
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger, make_run_dir
from src.utils.seed import set_seed


@torch.no_grad()
def evaluate(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    num_labels: int,
    logger: logging.Logger,
) -> tuple[dict[str, float], np.ndarray, float]:
    """Runs evaluation loop and computes metrics.
    
    Args:
        model: The sequence classification model to evaluate.
        dataloader: DataLoader for the evaluation dataset.
        device: Device to run the evaluation on.
        num_labels: Number of unique labels in the classification task.
        logger: Logger for logging evaluation progress.
    
    Returns:
        Computed metrics (accuracy, recall, precision, f1).
        Confusion matrix.
        Average loss over the evaluated dataset."""
    model.eval()

    # Initialization
    all_preds = []
    all_labels = []
    num_steps = len(dataloader)
    epoch_loss = 0.0

    for batch in tqdm(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            # Forward pass
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            loss = outputs.loss
            epoch_loss += loss.item()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute the metrics and the confusion matrix
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    metrics, cm = compute_metrics(y_true, y_pred, num_labels)

    return metrics, cm, epoch_loss / num_steps


def save_results(
    run_dir: str | Path,
    config: dict[str, any],
    labels: list[str],
    metrics: dict[str, float],
    confusion_matrix: np.ndarray,
    ) -> None:
    """Saves config and evaluation.
    
    Args:
        run_dir: Directory to save the results in.
        config: Configuration dictionary used for evaluation.
        labels: List of label names.
        metrics: Dictionary of computed metrics.
        confusion_matrix: Confusion matrix as a numpy array.
    """
    run_dir = Path(run_dir)

    enriched_config = {**config, 'labels': labels}
    with open(run_dir / "config_with_labels.json", "w", encoding="utf-8") as f:
        json.dump(enriched_config, f, indent=2, sort_keys=True)
    
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    
    np.save(run_dir / "confusion_matrix.npy", confusion_matrix)


def main(
    config_path: str,
    checkpoint: str = "pretrained",
    training_run_dir: str | None = None,
) -> None:
    """High-level evaluation entry point."""
    # Get the configuration
    overrides = {
        ("eval", "checkpoint"): checkpoint,
        ("eval", "training_run_dir"): training_run_dir,
    }
    cfg = load_yaml_config(config_path, overrides)

    # Set the seed
    set_seed(cfg['random']['seed'])

    # Set run_name and run_dir
    if cfg['eval']['checkpoint'] != 'pretrained' and not cfg['eval']['training_run_dir']:
        raise ValueError("training_run_dir is required when checkpoint is not 'pretrained'.")
    run_name = (
        f"eval_{cfg['eval']['mode']}_"
        f"{cfg['data']['dataset_name'].split('/')[-1]}_"
        f"{cfg['model']['pretrained_model_name'].replace('/', '-')}"
    )
    run_dir = make_run_dir(base_dir=Path('results'), run_name=run_name)

    # Set up the logger
    logger = get_logger(name=run_name, log_file=run_dir / 'run.log')

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting evaluation...")

    # Instantiate the tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['pretrained_model_name'])

    # Prepare the dataloader
    logger.info("Preparing dataloader...")
    dataloader, labels = prepare_dataloader(
        dataset_name=cfg['data']['dataset_name'],
        split=cfg['eval']['split'],
        tokenizer=tokenizer,
        batch_size=cfg['eval']['batch_size'],
        max_length=cfg['data']['max_length'],
        shuffle=False,
        keep_text=cfg['eval']['keep_text'],
        num_proc=cfg['data']['num_proc'],
        num_workers=cfg['data']['num_workers'],
    )
    num_labels = len(labels)

    # Load the model
    logger.info("Loading model...")
    model = load_model(
        model_name=cfg['model']['pretrained_model_name'],
        num_labels=num_labels,
        device=device,
        checkpoint=cfg['eval']['checkpoint'],
        run_dir=Path(cfg['eval']['training_run_dir']) if cfg['eval']['training_run_dir'] else None,
    )

    # Run evaluation and compute metrics
    logger.info("Running evaluation...")
    metrics, cm, _ = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        num_labels=num_labels,
        logger=logger,
    )

    # Save the results
    logger.info("Saving results...")
    save_results(
        run_dir=run_dir,
        config=cfg,
        labels=labels,
        metrics=metrics,
        confusion_matrix=cm,
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a sequence classification model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
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
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.training_run_dir)
