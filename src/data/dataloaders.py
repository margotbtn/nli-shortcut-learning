from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding
from transformers.tokenization_utils_base import BatchEncoding
from datasets import load_dataset


def seed_worker(_worker_id: int) -> None:
    """Seed worker processes for reproducible data loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    shuffle: bool = False,
    keep_text: bool = False,
    num_proc: int = min(4, os.cpu_count() or 1),
    num_workers: int = 2,
    generator: torch.Generator | None = None,
    ) -> tuple[DataLoader, list[str]]:
    """Prepares the dataloader and labels list.
    
    Args:
        dataset_name: Name of the dataset to load.
        split: Dataset split to use (e.g., 'validation').
        tokenizer: Tokenizer for processing the dataset.
        batch_size: Batch size for the dataloader.
        max_length: Maximum sequence length for tokenization.
        shuffle: Whether to shuffle the data in the DataLoader.
        keep_text: Whether to keep the original text fields in the dataset.
        num_proc: Number of processes to use for dataset mapping and filtering.
        num_workers: Number of worker processes for the DataLoader.
    
    Returns:
        A tuple containing the DataLoader and list of labels.
    """
    # Load the dataset
    ds = load_dataset(dataset_name, split=split)

    # Rename label column if necessary
    if 'label' in ds.column_names and 'labels' not in ds.column_names:
        ds = ds.rename_column('label', 'labels')

    # Filter unlabeled examples
    if "labels" in ds.column_names:
        ds = ds.filter(lambda x: x["labels"] != -1, num_proc=num_proc)
    
    # Tokenize the dataset
    cols_to_remove = [c for c in ds.column_names if c != 'labels'] if not keep_text else []
    ds = ds.map(
        lambda x: tokenize_batch(x, tokenizer, max_length),
        batched=True,
        remove_columns=cols_to_remove,
        num_proc=num_proc,
    )

    # Set the format for PyTorch - don't include text columns if they were kept!
    columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in ds.column_names:
        columns.append("token_type_ids")
    ds.set_format(type="torch", columns=columns)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        ), \
            ds.features['labels'].names
