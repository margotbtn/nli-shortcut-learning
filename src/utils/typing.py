# src/data/variants.py
"""Shared type aliases used across the project."""
from __future__ import annotations

from typing import Literal


# ---- Typing ----

Checkpoint = Literal["pretrained", "latest", "best"]
InputView = Literal["pair", "hypothesis_only"]
EvalSet = Literal["standard", "anti_shortcut", "ood"]
