from __future__ import annotations

import numpy as np
from pathlib import Path
import json

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
) -> tuple[dict[str, float], np.ndarray]:
    """Runs evaluation loop and computes metrics.
    
    Args:
        model: The sequence classification model to evaluate.
        dataloader: DataLoader for the evaluation dataset.
        device: Device to run the evaluation on.
        num_labels: Number of unique labels in the classification task.
    
    Returns:
        A tuple containing the computed metrics and confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            # Forward pass
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute the metrics and the confusion matrix
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    metrics, cm = compute_metrics(y_true, y_pred, num_labels)

    return metrics, cm


def save_results(
    output_path: str | Path,
    dataset_name: str,
    split: str,
    evaluation_mode: str,
    model_name: str,
    checkpoint: str,
    num_examples: int,
    labels: list[str],
    metrics: dict[str, float],
    confusion_matrix: np.ndarray,
    ) -> None:
    """Saves evaluation results to the specified output path.
    
    Args:
        output_path: Path to save the evaluation results.
        dataset_name: Name of the evaluated dataset.
        split: Dataset split used for evaluation.
        evaluation_mode: Mode of evaluation (e.g., 'standard', 'hypothesis_only').
        model_name: Name of the evaluated model.
        checkpoint: Checkpoint identifier of the evaluated model.
        num_examples: Number of examples evaluated.
        labels: List of label names.
        metrics: Dictionary of computed metrics.
        confusion_matrix: Confusion matrix as a numpy array.
    """
    to_save = {
        "evaluation_metadata": {
            "dataset": dataset_name,
            "split": split,
            "evaluation_mode": evaluation_mode,
            "model_name": model_name,
            "checkpoint": checkpoint,
            "num_examples": num_examples,
        },
        "labels": labels,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix.tolist(),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2, sort_keys=True)


def main() -> None:
    """High-level evaluation entry point."""
    # Configuration and setup
    cfg = load_yaml_config()
    set_seed(cfg['random']['seed'])
    run_name = (
        f"eval_{cfg['eval']['mode']}_"
        f"{cfg['data']['dataset_name'].split('/')[-1]}_"
        f"{cfg['model']['pretrained_model_name'].replace('/', '-')}"
    )
    run_dir = make_run_dir(base_dir='results', run_name=run_name)
    logger = get_logger(name=run_name, log_file=run_dir / 'run.log')
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
    )
    num_labels = len(labels)

    # Load the model
    logger.info("Loading model...")
    model = load_model(
        model_name=cfg['model']['pretrained_model_name'],
        num_labels=num_labels,
        device=device,
    )

    # Run evaluation and compute metrics
    logger.info("Running evaluation...")
    metrics, cm = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        num_labels=num_labels,
    )

    # Save the results
    logger.info("Saving results...")
    save_results(
        output_path=run_dir / 'evaluation_results.json',
        dataset_name=cfg['data']['dataset_name'],
        split=cfg['eval']['test_split'],
        evaluation_mode=cfg['eval']['mode'],
        model_name=cfg['model']['pretrained_model_name'],
        checkpoint='latest',
        num_examples=len(dataloader.dataset),
        labels=labels,
        metrics=metrics,
        confusion_matrix=cm,
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
