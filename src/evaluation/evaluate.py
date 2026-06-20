"""Phase 5 — unified evaluation: every model on the SAME test set.

## What this code does
This is the centerpiece of the benchmark. It loads the three trained models
(SVM, XGBoost, CNN), runs each on the identical held-out test split, and produces
one comparison table with:
- accuracy and macro-F1 (overall quality),
- per-class F1 (where each model is strong/weak by genre),
- inference latency in ms/clip (how expensive each model is to serve).

Latency is measured **model-only on CPU** (i.e. given the prepared input), one
clip at a time — the serving-realistic view, and the only way classic vs CNN is
an apples-to-apples cost comparison (their feature pre-processing differs).

Outputs: prints the table and writes `reports/comparison.csv`,
`reports/comparison.md`, and a confusion matrix per model.

    python -m src.evaluation.evaluate
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from src.data.manifest import label_maps
from src.evaluation.metrics import compute_metrics, save_confusion_matrix
from src.features.classic_features import feature_names
from src.utils import PROJECT_ROOT, load_config

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ----------------------------------------------------------------------------
# Per-model evaluation
# ----------------------------------------------------------------------------
def _load_classic_test():
    """Return (X_test, y_test) from the cached classic feature table."""
    fcfg = load_config("features")
    df = pd.read_parquet(PROJECT_ROOT / fcfg["classic"]["cache_path"])
    test = df[df["split"] == "test"]
    cols = feature_names(fcfg["classic"])
    return test[cols].to_numpy(), test["label_idx"].to_numpy()


def eval_classic(name: str, class_names: list[str]) -> dict | None:
    path = MODELS_DIR / f"classic_{name}.joblib"
    if not path.exists():
        return None
    model = joblib.load(path)
    X, y = _load_classic_test()

    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred, class_names)

    # latency: one sample at a time, mean over the test set (after a warmup).
    model.predict(X[:1])
    t0 = time.perf_counter()
    for i in range(len(X)):
        model.predict(X[i : i + 1])
    metrics["latency_ms"] = (time.perf_counter() - t0) / len(X) * 1000
    save_confusion_matrix(y, y_pred, class_names, REPORTS_DIR / f"cm_{name}.png",
                          title=f"{name} — test confusion matrix")
    return metrics


def eval_segmented(class_names: list[str]) -> dict | None:
    """Evaluate the 3s-segment SVM at clip level (average segment probabilities)."""
    path = MODELS_DIR / "classic_svm_segmented.joblib"
    fcfg = load_config("features")
    table = PROJECT_ROOT / fcfg["classic"]["segment_cache_path"]
    if not (path.exists() and table.exists()):
        return None

    from src.evaluation.aggregate import aggregate_proba_by_clip

    model = joblib.load(path)
    df = pd.read_parquet(table)
    test = df[df["split"] == "test"]
    cols = feature_names(fcfg["classic"])

    proba = model.predict_proba(test[cols].to_numpy())
    _, clip_proba, clip_true = aggregate_proba_by_clip(
        test["clip_id"].to_numpy(), proba, test["label_idx"].to_numpy()
    )
    clip_pred = clip_proba.argmax(1)
    metrics = compute_metrics(clip_true, clip_pred, class_names)

    # latency: per-clip = predict on its segments + average (model-only).
    clip_ids = list(dict.fromkeys(test["clip_id"]))
    Xc = {c: test[test.clip_id == c][cols].to_numpy() for c in clip_ids}
    model.predict_proba(Xc[clip_ids[0]])
    t0 = time.perf_counter()
    for c in clip_ids:
        model.predict_proba(Xc[c]).mean(axis=0)
    metrics["latency_ms"] = (time.perf_counter() - t0) / len(clip_ids) * 1000
    save_confusion_matrix(clip_true, clip_pred, class_names,
                          REPORTS_DIR / "cm_svm_segmented.png",
                          title="SVM (3s segments) — test confusion matrix")
    return metrics


def eval_embeddings(class_names: list[str], weights: str = "embeddings_logreg.joblib",
                    cm_name: str = "cm_panns_logreg.png") -> dict | None:
    """Evaluate a PANNs-embedding classifier (one embedding per clip)."""
    path = MODELS_DIR / weights
    table = PROJECT_ROOT / load_config("embeddings")["panns"]["cache_path"]
    if not (path.exists() and table.exists()):
        return None

    model = joblib.load(path)
    df = pd.read_parquet(table)
    test = df[df["split"] == "test"]
    cols = [c for c in df.columns if c.startswith("emb_")]
    X, y = test[cols].to_numpy(), test["label_idx"].to_numpy()

    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred, class_names)

    model.predict(X[:1])
    t0 = time.perf_counter()
    for i in range(len(X)):
        model.predict(X[i : i + 1])
    metrics["latency_ms"] = (time.perf_counter() - t0) / len(X) * 1000
    save_confusion_matrix(y, y_pred, class_names, REPORTS_DIR / cm_name,
                          title=f"{weights} — test confusion matrix")
    return metrics


def _clip_proba(table_path, col_filter, model):
    """Return {clip_id: proba_vector} and {clip_id: label} for test rows.

    `col_filter(name) -> bool` picks the feature columns, so the table is read
    only once (no extra pass just to discover column names).
    """
    df = pd.read_parquet(table_path)
    cols = [c for c in df.columns if col_filter(c)]
    test = df[df["split"] == "test"]
    proba = model.predict_proba(test[cols].to_numpy())
    ids = test["clip_id"].to_numpy()
    labels = test["label_idx"].to_numpy()
    return ({cid: proba[i] for i, cid in enumerate(ids)},
            {cid: labels[i] for i, cid in enumerate(ids)})


def eval_ensemble(class_names: list[str]) -> dict | None:
    """Soft-vote ensemble of the SVM (librosa) and PANNs probe, per clip."""
    svm_path = MODELS_DIR / "classic_svm.joblib"
    emb_path = MODELS_DIR / "embeddings_logreg.joblib"
    fcfg, ecfg = load_config("features"), load_config("embeddings")
    classic_tbl = PROJECT_ROOT / fcfg["classic"]["cache_path"]
    emb_tbl = PROJECT_ROOT / ecfg["panns"]["cache_path"]
    if not (svm_path.exists() and emb_path.exists() and classic_tbl.exists() and emb_tbl.exists()):
        return None

    classic_cols = set(feature_names(fcfg["classic"]))
    svm_p, labels = _clip_proba(classic_tbl, classic_cols.__contains__, joblib.load(svm_path))
    emb_p, _ = _clip_proba(emb_tbl, lambda c: c.startswith("emb_"), joblib.load(emb_path))

    clips = [c for c in svm_p if c in emb_p]
    y_true = np.array([labels[c] for c in clips])
    y_pred = np.array([np.argmax((svm_p[c] + emb_p[c]) / 2) for c in clips])
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["latency_ms"] = float("nan")  # sum of two models' costs; not measured
    save_confusion_matrix(y_true, y_pred, class_names, REPORTS_DIR / "cm_ensemble.png",
                          title="Ensemble (SVM + PANNs) — test confusion matrix")
    return metrics


def eval_cnn(class_names: list[str]) -> dict | None:
    weights = MODELS_DIR / "cnn.pt"
    arch = MODELS_DIR / "cnn_arch.json"
    if not (weights.exists() and arch.exists()):
        return None

    import torch  # local import: only the CNN path needs torch

    from src.data.melspec_dataset import MelspecDataset
    from src.models.cnn import build_cnn

    model = build_cnn(json.loads(arch.read_text()))
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()  # CPU — serving-realistic latency

    ds = MelspecDataset("test")
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in ds:
            logits = model(x.unsqueeze(0))  # add batch dim
            y_pred.append(int(logits.argmax(1)))
            y_true.append(int(y))
    metrics = compute_metrics(np.array(y_true), np.array(y_pred), class_names)

    # latency: single-clip forward pass, mean over the test set (after warmup).
    with torch.no_grad():
        x0, _ = ds[0]
        model(x0.unsqueeze(0))
        t0 = time.perf_counter()
        for i in range(len(ds)):
            xi, _ = ds[i]
            model(xi.unsqueeze(0))
    metrics["latency_ms"] = (time.perf_counter() - t0) / len(ds) * 1000
    save_confusion_matrix(np.array(y_true), np.array(y_pred), class_names,
                          REPORTS_DIR / "cm_cnn.png", title="CNN — test confusion matrix")
    return metrics


# ----------------------------------------------------------------------------
# Table assembly
# ----------------------------------------------------------------------------
def build_comparison_table(results: dict[str, dict], class_names: list[str]) -> pd.DataFrame:
    """Turn {model_name: metrics_dict} into a tidy, ordered comparison DataFrame."""
    headline = ["accuracy", "macro_f1", "latency_ms"]
    per_class = [f"f1_{c}" for c in class_names]
    rows = []
    for model_name, m in results.items():
        row = {"model": model_name}
        row.update({k: m[k] for k in headline})
        row.update({k: m[k] for k in per_class})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    # Rank by macro-F1 (the fairest single number on a balanced 10-way task).
    return df.sort_values("macro_f1", ascending=False)


def main() -> None:
    _, idx_to_genre = label_maps()
    class_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]

    results: dict[str, dict] = {}
    for name in ("svm", "xgboost"):
        m = eval_classic(name, class_names)
        if m:
            results[name] = m
    # tuned XGBoost (Optuna), if present
    xgb_tuned = eval_classic("xgboost_tuned", class_names)
    if xgb_tuned:
        results["xgboost_tuned"] = xgb_tuned
    seg = eval_segmented(class_names)
    if seg:
        results["svm_3s"] = seg
    emb = eval_embeddings(class_names)
    if emb:
        results["panns_logreg"] = emb
    emb_tuned = eval_embeddings(class_names, "embeddings_logreg_tuned.joblib",
                                "cm_panns_logreg_tuned.png")
    if emb_tuned:
        results["panns_logreg_tuned"] = emb_tuned
    ens = eval_ensemble(class_names)
    if ens:
        results["ensemble_svm_panns"] = ens
    cnn = eval_cnn(class_names)
    if cnn:
        results["cnn"] = cnn

    if not results:
        raise SystemExit(
            "No trained models found in models/. Train them first:\n"
            "  python -m src.training.train_classic\n"
            "  python -m src.training.train_cnn --device cuda"
        )

    df = build_comparison_table(results, class_names)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS_DIR / "comparison.csv")
    (REPORTS_DIR / "comparison.md").write_text(df.round(3).to_markdown())

    pd.set_option("display.width", 140, "display.max_columns", 20)
    print("\n=== Model comparison (same test set, ranked by macro-F1) ===")
    print(df[["accuracy", "macro_f1", "latency_ms"]].round(3).to_string())
    print(f"\nFull per-class table written to {REPORTS_DIR / 'comparison.csv'}")


if __name__ == "__main__":
    main()
