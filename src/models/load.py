from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification


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