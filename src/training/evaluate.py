from __future__ import annotations

import numpy as np
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase, \
    AutoModelForSequenceClassification, default_data_collator
from transformers.tokenization_utils_base import BatchEncoding
from datasets import load_dataset

from src.training.metrics import compute_metrics
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger, make_run_dir
from src.utils.seed import set_seed


def tokenize_batch(
        batch: dict[str, list[str]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        ) -> BatchEncoding:
    """Tokenizes a batch of data using the provided tokenizer.

    Args:
        batch: A dictionary containing the batch data with keys 'premise' and 'hypothesis'.
        tokenizer: The tokenizer to use for tokenization.
        max_length: Maximum sequence length for tokenization.
    
    Returns:
        A BatchEncoding containing tokenized fields (e.g., input_ids, attention_mask).
        Values are typically Python lists at this stage; they become torch.Tensors after
        calling ds.set_format(type="torch").
    """
    return tokenizer(
        batch['premise'],
        batch['hypothesis'],
        padding=False,
        truncation=True,
        max_length=max_length,
    )


def prepare_dataloader(
    dataset_name: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int,
    num_proc: int | None = None,
    ) -> tuple[DataLoader, list[str]]:
    """Prepares the evaluation dataloader and label list.
    
    Args:
        dataset_name: Name of the dataset to load.
        split: Dataset split to use (e.g., 'validation').
        tokenizer: Tokenizer for processing the dataset.
        batch_size: Batch size for the dataloader.
        max_length: Maximum sequence length for tokenization.
        num_proc: Number of processes to use for dataset mapping and filtering.
    
    Returns:
        A tuple containing the DataLoader and list of labels.
    """
    # Load the dataset
    ds = load_dataset(dataset_name, split=split)

    # Filter unlabeled examples
    if "label" in ds.column_names:
        ds = ds.filter(lambda x: x["label"] != -1, num_proc=num_proc)
    
    # Tokenize the dataset
    ds = ds.map(
        lambda x: tokenize_batch(x, tokenizer, max_length),
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in ("label",)],
        num_proc=num_proc,
    )

    # Set the format for PyTorch
    ds.set_format(type="torch")

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator), \
        ds.features['label'].names


def load_model(model_name: str, num_labels: int, device: torch.device) -> AutoModelForSequenceClassification:
    """Loads a sequence classification model on the target device.
    
    Args:
        model_name: Pretrained model name or path.
        num_labels: Number of labels for classification.
        device: Target device for the model.
    
    Returns:
        The loaded model on the specified device."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    model.to(device)
    return model


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
            labels = batch.pop("label")

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
        split=cfg['eval']['split'],
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
