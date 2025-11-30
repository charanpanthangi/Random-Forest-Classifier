"""Run the full Random Forest classification pipeline on the Iris dataset."""
from pathlib import Path

import pandas as pd

from app.data import load_iris_dataset
from app.evaluate import confusion, evaluate_predictions
from app.model import build_model
from app.preprocess import split_data
from app.visualize import plot_confusion_matrix, plot_feature_importance, plot_pca_scatter


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_pipeline() -> None:
    """Train, evaluate, and visualize a Random Forest classifier."""
    print("\n🔎 Loading Iris dataset...")
    X, y = load_iris_dataset()

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Building RandomForestClassifier...")
    model = build_model()

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)

    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name.capitalize()}: {value:.3f}")

    feature_names = X.columns
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        by="importance", ascending=False
    )
    print("\nTop feature importances:")
    print(importance_df.to_string(index=False))

    conf_matrix = confusion(y_test, y_pred)
    class_names = [str(label) for label in sorted(y.unique())]

    cm_path = plot_confusion_matrix(conf_matrix, class_names)
    fi_path = plot_feature_importance(feature_names, importances)
    pca_path = plot_pca_scatter(X.values, y.values, class_names)

    print("\nSaved visualizations:")
    for path in [cm_path, fi_path, pca_path]:
        print(f"  - {path}")


if __name__ == "__main__":
    run_pipeline()
