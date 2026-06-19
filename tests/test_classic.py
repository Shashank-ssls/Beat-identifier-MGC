"""Tests for the classic-ML path (Phase 3): model construction + metrics.

Fast and dataset-free — they use tiny synthetic data so CI stays quick and
doesn't need GTZAN or the feature cache.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import compute_metrics
from src.models.classic import build_model
from src.utils import load_config

CCFG = load_config("classic")
CLASS_NAMES = [f"g{i}" for i in range(3)]


def _toy_data(n=60, n_features=8, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    # Make classes linearly separable-ish so a quick fit actually learns.
    y = rng.integers(0, n_classes, size=n)
    X = rng.normal(size=(n, n_features)) + y[:, None]
    return X, y


@pytest.mark.parametrize("name", ["xgboost", "svm"])
def test_build_model_is_pipeline_with_scaler(name):
    model = build_model(name, CCFG[name], seed=42)
    assert isinstance(model, Pipeline)
    assert model.steps[0][0] == "scaler"  # scaler must come first (anti-leakage)


@pytest.mark.parametrize("name", ["xgboost", "svm"])
def test_model_fits_and_predicts(name):
    X, y = _toy_data(n_classes=3)
    model = build_model(name, CCFG[name], seed=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_build_model_rejects_unknown():
    with pytest.raises(ValueError):
        build_model("randomforest", {}, seed=42)


def test_compute_metrics_keys_and_ranges():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    m = compute_metrics(y_true, y_pred, CLASS_NAMES)
    assert {"accuracy", "macro_f1", "f1_g0", "f1_g1", "f1_g2"} <= set(m)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0
