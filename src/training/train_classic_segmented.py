"""Train an SVM on 3s segments with a GroupKFold hyperparameter sweep.

## What this code does
The accuracy-boosting variant of the classic path:
1. Loads the segmented feature table (~10x rows).
2. Runs a grid search over SVM C/gamma using **GroupKFold grouped by clip_id** —
   so a clip's segments never straddle a CV fold's train/val boundary (leakage-
   safe model selection).
3. Refits the best pipeline on all training segments.
4. Evaluates on val and test at **clip level** (averaging each clip's segment
   probabilities), so the number is comparable to every other model in the
   benchmark (same 150 test clips).
5. Logs to MLflow and saves the model + segment metadata for serving.

    python -m src.training.train_classic_segmented
"""

from __future__ import annotations

import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.data.manifest import label_maps
from src.evaluation.aggregate import aggregate_proba_by_clip
from src.evaluation.metrics import compute_metrics, save_confusion_matrix
from src.features.classic_features import feature_names
from src.models.classic import build_model
from src.utils import PROJECT_ROOT, load_config, set_seed

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def _clip_level_metrics(model, df_split, cols, class_names):
    """Predict segments, average to clip level, return (metrics, true, pred)."""
    proba = model.predict_proba(df_split[cols].to_numpy())
    _, clip_proba, clip_true = aggregate_proba_by_clip(
        df_split["clip_id"].to_numpy(), proba, df_split["label_idx"].to_numpy()
    )
    clip_pred = clip_proba.argmax(1)
    return compute_metrics(clip_true, clip_pred, class_names), clip_true, clip_pred


def main() -> None:
    ccfg = load_config("classic")
    fcfg = load_config("features")
    set_seed(ccfg["seed"])

    table = PROJECT_ROOT / fcfg["classic"]["segment_cache_path"]
    if not table.exists():
        raise SystemExit(
            f"Segmented table not found at {table}. "
            "Run `python -m src.features.extract_segments` first."
        )
    df = pd.read_parquet(table)
    cols = feature_names(fcfg["classic"])
    _, idx_to_genre = label_maps()
    class_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]

    train = df[df["split"] == "train"]
    X_tr, y_tr, groups = train[cols].to_numpy(), train["label_idx"].to_numpy(), train["clip_id"].to_numpy()
    print(f"Segments: train={len(train)} val={int((df.split=='val').sum())} test={int((df.split=='test').sum())}")

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(ccfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="svm_segmented"):
        # --- GroupKFold grid search (leakage-safe selection) ---
        sweep = ccfg["svm_sweep"]
        base = build_model("svm", ccfg["svm"], ccfg["seed"])
        search = GridSearchCV(
            base, sweep["param_grid"],
            cv=GroupKFold(n_splits=sweep["cv_folds"]),
            scoring="f1_macro", n_jobs=-1,
        )
        search.fit(X_tr, y_tr, groups=groups)
        best = search.best_estimator_
        print(f"  best params: {search.best_params_}  (cv macroF1={search.best_score_:.3f})")

        # --- clip-level eval on val + test ---
        val_m, _, _ = _clip_level_metrics(best, df[df.split == "val"], cols, class_names)
        test_m, t_true, t_pred = _clip_level_metrics(best, df[df.split == "test"], cols, class_names)

        mlflow.log_params({k: v for k, v in search.best_params_.items()})
        mlflow.log_param("segment_seconds", fcfg["classic"]["segment_seconds"])
        mlflow.log_param("cv_macro_f1", round(search.best_score_, 4))
        mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})

        cm = save_confusion_matrix(t_true, t_pred, class_names,
                                   REPORTS_DIR / "cm_svm_segmented.png",
                                   title="SVM (3s segments) — test confusion matrix")
        mlflow.log_artifact(str(cm), artifact_path="plots")
        mlflow.sklearn.log_model(best, artifact_path="classic_svm_segmented")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, MODELS_DIR / "classic_svm_segmented.joblib")
        (MODELS_DIR / "classic_svm_segmented_meta.json").write_text(
            json.dumps({"segment_seconds": fcfg["classic"]["segment_seconds"], "kind": "classic_segmented"}, indent=2)
        )

    print(f"\n  SVM(3s)  test_acc={test_m['accuracy']:.3f}  test_macroF1={test_m['macro_f1']:.3f}")
    print(f"  (was: SVM(30s) acc=0.807)  saved -> {MODELS_DIR / 'classic_svm_segmented.joblib'}")


if __name__ == "__main__":
    main()
