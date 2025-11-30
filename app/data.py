"""Data loading utilities for the Random Forest classifier demo."""
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris


def load_iris_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the Iris dataset as feature and target data.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.
    """

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y
