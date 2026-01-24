from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_labels: int
    ) -> tuple[dict[str, float], np.ndarray]:
    """Computes evaluation metrics and confusion metrics for classification.

    List of the computed metrics:
        - accuracy
        - f1_score (macro)
        - recall (macro)
        - precision (macro)

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        num_labels: Number of unique labels in the classification task.
    
    Returns:
        Dictionary mapping metric names to values.
        Confusion matrix as a 2D numpy array.
    """
    # Compute specified metrics
    results: dict[str, float] = {}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['f1_macro'] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results['recall_macro'] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    results['precision_macro'] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Compute confusion metrics
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))

    return results, cm
