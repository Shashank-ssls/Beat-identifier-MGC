"""Train a linear probe on the frozen PANNs embeddings.

## What this code does
Transfer learning, the simple way: take the 2048-dim PANNs embeddings (already
rich because the model was pretrained on AudioSet) and fit a plain
`StandardScaler -> LogisticRegression` on top. Because the embeddings do the
heavy lifting, even this tiny classifier beats the hand-feature SVM and the
from-scratch CNN. One embedding per clip → no segment aggregation needed.

    python -m src.training.train_embeddings
"""

from __future__ import annotations

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.manifest import label_maps
from src.evaluation.metrics import compute_metrics, save_confusion_matrix
from src.utils import PROJECT_ROOT, load_config, set_seed

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def embedding_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def main() -> None:
    ecfg = load_config("embeddings")
    set_seed(ecfg["seed"])

    table = PROJECT_ROOT / ecfg["panns"]["cache_path"]
    if not table.exists():
        raise SystemExit(
            f"Embeddings table not found at {table}. "
            "Run `python -m src.features.extract_embeddings` first."
        )
    df = pd.read_parquet(table)
    cols = embedding_cols(df)
    _, idx_to_genre = label_maps()
    class_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]

    def split_xy(name):
        s = df[df["split"] == name]
        return s[cols].to_numpy(), s["label_idx"].to_numpy()

    X_tr, y_tr = split_xy("train")
    X_val, y_val = split_xy("val")
    X_te, y_te = split_xy("test")
    print(f"Embeddings: train={len(y_tr)} val={len(y_val)} test={len(y_te)} dim={len(cols)}")

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(ecfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="panns_logreg"):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=ecfg["classifier"]["C"],
                max_iter=ecfg["classifier"]["max_iter"],
                random_state=ecfg["seed"],
            )),
        ])
        clf.fit(X_tr, y_tr)

        val_m = compute_metrics(y_val, clf.predict(X_val), class_names)
        te_pred = clf.predict(X_te)
        test_m = compute_metrics(y_te, te_pred, class_names)

        mlflow.log_params({"backbone": "panns_cnn14", "embedding_dim": len(cols),
                           "C": ecfg["classifier"]["C"]})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        cm = save_confusion_matrix(y_te, te_pred, class_names,
                                   REPORTS_DIR / "cm_panns_logreg.png",
                                   title="PANNs + logreg — test confusion matrix")
        mlflow.log_artifact(str(cm), artifact_path="plots")
        mlflow.sklearn.log_model(clf, artifact_path="panns_logreg")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, MODELS_DIR / "embeddings_logreg.joblib")

    print(f"\n  PANNs+logreg  test_acc={test_m['accuracy']:.3f}  test_macroF1={test_m['macro_f1']:.3f}")
    print("  (baselines: SVM 0.807, XGBoost 0.713, CNN 0.693)")


if __name__ == "__main__":
    main()
