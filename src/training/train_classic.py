"""Train the classic-ML models (XGBoost + SVM) and log to MLflow.

## What this code does
1. Loads the cached classic feature table (parquet) and splits it into
   train/val/test using the committed split column — the SAME split every model
   sees, so the benchmark is fair.
2. For each model: fits `StandardScaler -> estimator` on TRAIN only, evaluates
   on VAL (model-selection view) and TEST (final), and logs everything to
   MLflow: hyperparameters, metrics, a confusion-matrix image, and the fitted
   model itself.
3. Saves the fitted pipeline to `models/` so Phase 5 (unified eval) and Phase 6
   (serving) can load it without retraining.

## Where it runs
On Kaggle you'd run the same code and then **export the artifacts back to the
repo**: copy `mlruns/` (or just the saved `.joblib` files) from the Kaggle output
into the repo, and they register in your LOCAL MLflow. Locally (our case) it
writes straight to `./mlruns`.

    python -m src.training.train_classic              # both models
    python -m src.training.train_classic --model svm  # just one
"""

from __future__ import annotations

import argparse

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.data.manifest import label_maps
from src.evaluation.metrics import compute_metrics, save_confusion_matrix
from src.features.classic_features import feature_names
from src.models.classic import build_model
from src.utils import PROJECT_ROOT, load_config, set_seed

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def _load_split_frames(feature_cols: list[str]):
    """Return (X, y) dicts keyed by split, read from the parquet feature table."""
    fcfg = load_config("features")
    table_path = PROJECT_ROOT / fcfg["classic"]["cache_path"]
    if not table_path.exists():
        raise FileNotFoundError(
            f"Feature table not found at {table_path}. "
            "Run `python -m src.features.extract --what classic` first."
        )
    df = pd.read_parquet(table_path)

    X, y = {}, {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        X[split] = sub[feature_cols].to_numpy()
        y[split] = sub["label_idx"].to_numpy()
    return X, y


def train_one(name: str, X, y, class_names: list[str]) -> dict:
    """Train+evaluate one model, log to MLflow, persist it. Returns test metrics."""
    ccfg = load_config("classic")
    seed = ccfg["seed"]

    mlflow.set_experiment(ccfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=name):
        model = build_model(name, ccfg[name], seed)
        model.fit(X["train"], y["train"])

        # Evaluate on val (selection) and test (final).
        val_metrics = compute_metrics(y["val"], model.predict(X["val"]), class_names)
        test_pred = model.predict(X["test"])
        test_metrics = compute_metrics(y["test"], test_pred, class_names)

        # --- MLflow logging ---
        mlflow.log_params({f"{name}__{k}": v for k, v in ccfg[name].items()})
        mlflow.log_param("n_features", X["train"].shape[1])
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        cm_path = save_confusion_matrix(
            y["test"], test_pred, class_names,
            REPORTS_DIR / f"cm_classic_{name}.png",
            title=f"{name} — test confusion matrix",
        )
        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        mlflow.sklearn.log_model(model, artifact_path=f"classic_{name}")

        # --- Persist locally for Phase 5/6 ---
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / f"classic_{name}.joblib")

    print(
        f"  {name:8s}  test_acc={test_metrics['accuracy']:.3f}  "
        f"test_macroF1={test_metrics['macro_f1']:.3f}"
    )
    return test_metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train classic-ML models with MLflow.")
    p.add_argument("--model", choices=["xgboost", "svm", "both"], default="both")
    args = p.parse_args()

    set_seed(load_config("classic")["seed"])
    mlflow.set_tracking_uri((MLRUNS_DIR).as_uri())

    _, idx_to_genre = label_maps()
    class_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]

    feature_cols = feature_names(load_config("features")["classic"])
    X, y = _load_split_frames(feature_cols)
    print(f"Loaded features: train={len(y['train'])} val={len(y['val'])} test={len(y['test'])}")

    models = ["xgboost", "svm"] if args.model == "both" else [args.model]
    for name in models:
        train_one(name, X, y, class_names)

    print(f"\nMLflow runs logged to {MLRUNS_DIR}")
    print("View with:  mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
