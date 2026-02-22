# src/models/load.py
"""Utilities for loading pretrained and fine-tuned sequence classification models onto a target device."""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Literal
from transformers import AutoModelForSequenceClassification


def load_model(
        model_name: str,
        num_labels: int,
        device: torch.device,
        checkpoint: Literal['pretrained', 'latest', 'best'] = 'pretrained',
        run_dir: str | Path | None = None,
        ) -> AutoModelForSequenceClassification:
    """Loads a sequence classification model on the target device.
    
    Args:
        model_name: Pretrained model name.
        num_labels: Number of labels for classification.
        device: Target device for the model.
        checkpoint: Checkpoint type to load.
        run_dir: Directory of the training run (required if checkpoint is not 'pretrained').
    
    Returns:
        The loaded model on the specified device."""
    # Validate arguments
    if checkpoint != 'pretrained':
        if run_dir is None:
            raise ValueError("run_dir must be provided when loading a checkpoint other than 'pretrained'.")
        else:
            run_dir = Path(run_dir)
            if not run_dir.exists():
                raise ValueError(f"Provided run_dir '{run_dir}' does not exist.")
    
    # Load the model
    model_name_or_path = model_name if checkpoint == 'pretrained' else run_dir / 'checkpoints' / checkpoint / 'model'
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
    )

    # Move model to device
    model.to(device)
    
    return model
