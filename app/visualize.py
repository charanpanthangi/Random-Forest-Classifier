"""Plotting helpers for evaluation and feature exploration."""
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay


sns.set(style="whitegrid")


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str]) -> Path:
    """Save a confusion matrix heatmap as SVG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    svg_path = OUTPUT_DIR / "confusion_matrix.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return svg_path


def plot_feature_importance(feature_names: Sequence[str], importances: Sequence[float]) -> Path:
    """Save feature importance bar plot as SVG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=list(importances), y=list(feature_names), ax=ax, palette="viridis")
    ax.set_title("Feature Importance (higher means more predictive)")
    ax.set_xlabel("Importance score")
    svg_path = OUTPUT_DIR / "feature_importance.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return svg_path


def plot_pca_scatter(X: np.ndarray, y: np.ndarray, class_names: Sequence[str]) -> Path:
    """Project features into 2D with PCA and save scatter plot."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", edgecolor="k")
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_title("PCA projection of Iris features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    svg_path = OUTPUT_DIR / "pca_scatter.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return svg_path
