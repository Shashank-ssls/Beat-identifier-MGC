"""Shared evaluation metrics + confusion-matrix plotting.

## What this code does
One place that turns predictions into the numbers we report for EVERY model
(classic and CNN), so the benchmark compares apples to apples. We report:
- accuracy: fraction correct overall.
- macro-F1: F1 averaged equally across the 10 genres (so a model can't win just
  by nailing the easy classes — important on a balanced 10-way problem).
- per-class F1: where each genre individually is strong or weak.
Plus a confusion-matrix image showing which genres get mistaken for which.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI — we only save PNGs (works on Kaggle/CI too)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    """Return a flat dict of metrics suitable for logging to MLflow."""
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    for name, f1 in zip(class_names, per_class):
        metrics[f"f1_{name}"] = float(f1)
    return metrics


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str = "Confusion matrix",
) -> Path:
    """Render a normalised confusion matrix to a PNG and return its path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        labels=range(len(class_names)),
        normalize="true",        # row-normalised → per-genre recall on the diagonal
        cmap="Blues",
        ax=ax,
        colorbar=False,
        values_format=".2f",
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
