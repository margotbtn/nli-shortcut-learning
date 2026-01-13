from __future__ import annotations

from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator
from transformers.tokenization_utils_base import BatchEncoding
from datasets import load_dataset

from src.training.metrics import compute_metrics
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def tokenize_batch(
        batch: Dict[str, List[str]],
        tokenizer: AutoTokenizer,
        max_length: int,
        ) -> BatchEncoding:
    """Tokenizes a batch of data using the provided tokenizer.

    Args:
        batch: A dictionary containing the batch data with keys 'premise' and 'hypothesis'.
        tokenizer: The tokenizer to use for tokenization.
        max_length: Maximum sequence length for tokenization.
    
    Returns:
        A dictionary containing tokenized inputs as torch Tensors.
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
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    ) -> Tuple[DataLoader, List[str]]:
    """Prepares the evaluation dataloader and label list.
    
    Args:
        dataset_name: Name of the dataset to load.
        split: Dataset split to use (e.g., 'validation').
        tokenizer: Tokenizer for processing the dataset.
        batch_size: Batch size for the dataloader.
        max_length: Maximum sequence length for tokenization.
    
    Returns:
        A tuple containing the DataLoader and list of labels.
    """
    ds = load_dataset(dataset_name, split=split)
    ds = ds.map(
        lambda x: tokenize_batch(x, tokenizer, max_length),
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in ("label",)]
    )
    ds.set_format(type="torch")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator), ds.features['label'].names


def load_model(
    model_name: str,
    num_labels: int,
    device: torch.device,
) -> AutoModelForSequenceClassification:
    """Load a sequence classification model on the target device."""
    pass


@torch.no_grad()
def evaluate(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    num_labels: int,
) -> tuple[dict[str, float], torch.Tensor]:
    """Run evaluation loop and compute metrics."""
    pass


def main() -> None:
    """High-level evaluation entry point."""
    # Configuration and setup
    cfg = load_yaml_config()
    logger = get_logger(__name__)
    set_seed(cfg['random']['seed'])

    # Instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['pretrained_model_name'])

    # Prepare the dataloader
    dataloader, labels = prepare_dataloader(
        dataset_name=cfg['data']['dataset_name'],
        split = 'validation',
        tokenizer=tokenizer,
        batch_size=cfg['eval']['batch_size'],
        max_length=cfg['data']['max_length'],
    )


if __name__ == "__main__":
    main()
