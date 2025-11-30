"""Model evaluation utilities."""
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


Metrics = Dict[str, float]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Compute common classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary of metrics: accuracy, precision, recall, f1.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the confusion matrix."""

    return confusion_matrix(y_true, y_pred)
