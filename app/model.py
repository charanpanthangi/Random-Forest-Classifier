"""Model definition for the Random Forest classifier."""
from sklearn.ensemble import RandomForestClassifier


def build_model() -> RandomForestClassifier:
    """Create a RandomForestClassifier with sensible defaults.

    The model uses many decision trees (n_estimators) trained on bootstrap
    samples. Each tree sees a random subset of features (max_features), which
    increases diversity and helps the ensemble generalize better than a single
    tree.
    """

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
    )
