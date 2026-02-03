# src/data/standard.py
from __future__ import annotations

import os
from datasets import Dataset, load_dataset

from src.utils.typing import InputView


def preprocess_dataset(
        dataset_name: str,
        split: str,
        num_proc: int = min(4, os.cpu_count() or 1),
        input_view: InputView = "pair",
        ) ->Dataset:
    """Loads and applies basic preprocessing on the dataset.
    
    Args:
        dataset_name: Name of the dataset to load.
        split: Dataset split to use (e.g., 'validation').
        num_proc: Number of processes to use for dataset mapping and filtering.
        hypothesis_only: True if you want to keep only the hypothesis sentence.
    
    Returns:
        A HuggingFace preprocessed dataset.
    """
    # Load the dataset
    ds = load_dataset(dataset_name, split=split)

    # Rename label column if necessary
    if 'label' in ds.column_names and 'labels' not in ds.column_names:
        ds = ds.rename_column('label', 'labels')

    # Filter unlabeled examples
    if "labels" in ds.column_names:
        ds = ds.filter(lambda x: x["labels"] != -1, num_proc=num_proc)

    # Remove premise if required
    if input_view == 'hypothesis_only':
        ds = ds.remove_columns('premise')
    
    return ds