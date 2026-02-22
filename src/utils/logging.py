# src/utils/logging.py
"""Logging utilities for creating timestamped run directories and configured loggers."""
from __future__ import annotations

from typing import Any
import logging
from pathlib import Path
from datetime import datetime
import json


LOGGING_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET
}


def make_run_dir(base_dir: str | Path, run_name: str | None) -> Path:
    """Creates a timestamped run directory.
    Example: results/logs/2026-01-11_143210_baseline/

    Args:
        base_dir: Base directory where run directories are created.
        run_name: Optional name for the run.
    
    Returns:
        Path to the created run directory.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = "run"
    run_dir = base_dir / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{run_name.replace(' ', '_')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def get_logger(
        name: str = "nli",
        log_file: str | Path | None = None,
        level: str = "INFO"
        ) -> logging.Logger:
    """Creates a logger that logs to console and optionally to a file.

    Requirements:
    - No duplicate handlers if called multiple times.
    - Timestamped format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    - If log_file is provided, create parent directories.

    Args:
        name: Name of the logger.
        log_file: Optional path to a log file.
        level: Logging level as a string (e.g., "INFO", "DEBUG").
    
    Returns:
        Configured logger instance.
    """
    level = level.upper()
    if level not in LOGGING_LEVELS:
        raise ValueError(f"Unknown logging level: {level}")
    
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVELS[level])
    logger.propagate = False

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler to see logs in console
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOGGING_LEVELS[level])
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler to save logs to a file
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(LOGGING_LEVELS[level])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Saves a config dictionary to disk as JSON.

    Args:
        cfg: Configuration dictionary.
        path: Path to save the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
