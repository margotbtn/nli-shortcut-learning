# src/utils/seed.py
from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value.
        deterministic: Whether to enable deterministic PyTorch behavior.
    """
    # Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
