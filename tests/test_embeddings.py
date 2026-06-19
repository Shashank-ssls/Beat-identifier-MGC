"""Tests for the pretrained-embedding path (linear probe assembly).

Dataset-free: synthetic embeddings, so this runs without PANNs or the checkpoint.
The real extraction is exercised by `python -m src.features.extract_embeddings`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.train_embeddings import embedding_cols


def test_embedding_cols_selects_only_emb_columns():
    df = pd.DataFrame(
        {"clip_id": ["a"], "genre": ["blues"], "label_idx": [0],
         "split": ["train"], "emb_0": [0.1], "emb_1": [0.2]}
    )
    assert embedding_cols(df) == ["emb_0", "emb_1"]


def test_linear_probe_fits_on_synthetic_embeddings():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, size=120)
    X = rng.normal(size=(120, 64)) + y[:, None]  # separable-ish
    clf = Pipeline([("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=500))])
    clf.fit(X, y)
    assert clf.predict(X).shape == y.shape
    assert clf.score(X, y) > 0.5
