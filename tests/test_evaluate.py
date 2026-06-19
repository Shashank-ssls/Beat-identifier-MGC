"""Tests for unified evaluation table assembly (Phase 5).

Dataset-free: feeds synthetic per-model metric dicts to the table builder, so it
runs without trained models. The actual model evaluation is exercised by running
`python -m src.evaluation.evaluate` (covered manually, needs models/).
"""

from __future__ import annotations

from src.evaluation.evaluate import build_comparison_table

CLASS_NAMES = ["blues", "rock", "jazz"]


def _metrics(acc, f1, latency):
    m = {"accuracy": acc, "macro_f1": f1, "latency_ms": latency}
    for c in CLASS_NAMES:
        m[f"f1_{c}"] = f1
    return m


def test_table_has_all_models_and_columns():
    results = {
        "svm": _metrics(0.80, 0.80, 1.2),
        "cnn": _metrics(0.69, 0.67, 8.0),
    }
    df = build_comparison_table(results, CLASS_NAMES)
    assert set(df.index) == {"svm", "cnn"}
    for col in ["accuracy", "macro_f1", "latency_ms", "f1_blues", "f1_rock", "f1_jazz"]:
        assert col in df.columns


def test_table_ranked_by_macro_f1_desc():
    results = {
        "xgboost": _metrics(0.71, 0.71, 0.5),
        "svm": _metrics(0.80, 0.80, 1.2),
        "cnn": _metrics(0.69, 0.67, 8.0),
    }
    df = build_comparison_table(results, CLASS_NAMES)
    assert list(df.index) == ["svm", "xgboost", "cnn"]  # best macro-F1 first
