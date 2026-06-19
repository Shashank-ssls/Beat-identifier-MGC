"""Classic-ML model definitions: XGBoost and SVM.

## What this code does
Builds the two classic models as scikit-learn **Pipelines**. Each pipeline is
`StandardScaler -> estimator`. Wrapping the scaler INSIDE the pipeline is the
key anti-leakage detail: when we call `.fit(X_train)`, the scaler learns its
mean/std from the training data only, and the exact same transform is reused at
predict time. No test statistics ever leak into training.

- XGBoost: gradient-boosted trees. Strong on tabular features; usually the
  classic-path frontrunner on GTZAN.
- SVM (RBF kernel): finds a max-margin boundary in a transformed space. Needs
  scaled inputs (hence the scaler) and gives calibrated-ish probabilities.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_model(name: str, cfg: dict, seed: int) -> Pipeline:
    """Return a `StandardScaler -> estimator` pipeline for the named model.

    `cfg` is the relevant sub-dict from configs/classic.yaml.
    """
    if name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            objective=cfg["objective"],
            tree_method=cfg["tree_method"],
            random_state=seed,
            n_jobs=-1,
        )
    elif name == "svm":
        estimator = SVC(
            kernel=cfg["kernel"],
            C=cfg["C"],
            gamma=cfg["gamma"],
            probability=cfg["probability"],
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown classic model: {name!r}")

    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
