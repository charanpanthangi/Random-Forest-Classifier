"""Preprocessing helpers including train/test split.

Tree-based models like Random Forests are mostly scale-invariant, so we keep the
pipeline simple and do not apply feature scaling here. The goal is to highlight
ensemble learning concepts without extra preprocessing steps.
"""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and targets into train and test subsets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction to reserve for testing.
        random_state: Seed for reproducibility.

    Returns:
        Train/test splits for features and targets.
    """

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
