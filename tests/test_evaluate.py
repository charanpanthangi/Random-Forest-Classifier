import numpy as np

from app.evaluate import confusion, evaluate_predictions


def test_evaluate_predictions_returns_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 2, 0])
    matrix = confusion(y_true, y_pred)
    assert matrix.shape == (3, 3)
