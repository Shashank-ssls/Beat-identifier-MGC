"""Optuna hyperparameter tuning for the tabular models.

## What this code does
Searches for better hyperparameters with Optuna (Bayesian-ish search) instead of
hand-picked values, using **cross-validation on the training split only** (the
test set is never touched during tuning). Two targets:

- `xgboost` — tunes depth / learning-rate / estimators / subsampling on the
  classic librosa features.
- `probe`   — tunes the regularisation `C` of the logistic-regression probe on
  the frozen PANNs embeddings.

Best params + CV score are logged to MLflow; the refit best model is saved so
`evaluate` can pick it up.

    python -m src.training.tune --target xgboost --trials 40
    python -m src.training.tune --target probe   --trials 30
"""

from __future__ import annotations

import argparse

import joblib
import mlflow
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils import PROJECT_ROOT, load_config, set_seed

MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def _load(table_path, col_filter):
    df = pd.read_parquet(table_path)
    cols = [c for c in df.columns if col_filter(c)]
    tr = df[df["split"].isin(["train", "val"])]  # tune on train+val, never test
    return tr[cols].to_numpy(), tr["label_idx"].to_numpy(), cols


def tune_xgboost(trials: int, seed: int):
    fcfg = load_config("features")
    from src.features.classic_features import feature_names
    names = set(feature_names(fcfg["classic"]))
    X, y, _ = _load(PROJECT_ROOT / fcfg["classic"]["cache_path"], lambda c: c in names)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            objective="multi:softprob", tree_method="hist",
            random_state=seed, n_jobs=-1,
        )
        model = Pipeline([("scaler", StandardScaler()), ("clf", XGBClassifier(**params))])
        return cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", XGBClassifier(objective="multi:softprob", tree_method="hist",
                                            random_state=seed, n_jobs=-1,
                                            **study.best_params))])
    model.fit(X, y)
    joblib.dump(model, MODELS_DIR / "classic_xgboost_tuned.joblib")
    return study, "classic_xgboost_tuned.joblib"


def tune_probe(trials: int, seed: int):
    ecfg = load_config("embeddings")
    X, y, _ = _load(PROJECT_ROOT / ecfg["panns"]["cache_path"], lambda c: c.startswith("emb_"))
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        model = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed))])
        return cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, random_state=seed,
                                                 **study.best_params))])
    model.fit(X, y)
    joblib.dump(model, MODELS_DIR / "embeddings_logreg_tuned.joblib")
    return study, "embeddings_logreg_tuned.joblib"


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna tuning for tabular models.")
    p.add_argument("--target", choices=["xgboost", "probe"], required=True)
    p.add_argument("--trials", type=int, default=30)
    args = p.parse_args()

    seed = 42
    set_seed(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("genre-tuning")

    runner = {"xgboost": tune_xgboost, "probe": tune_probe}[args.target]
    with mlflow.start_run(run_name=f"tune_{args.target}"):
        study, saved = runner(args.trials, seed)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("cv_macro_f1", study.best_value)

    print(f"\n  best CV macro-F1: {study.best_value:.4f}")
    print(f"  best params     : {study.best_params}")
    print(f"  saved -> {MODELS_DIR / saved}")


if __name__ == "__main__":
    main()
