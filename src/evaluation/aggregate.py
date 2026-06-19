"""Aggregate segment-level predictions back to clip-level.

## What this code does
When a model is trained on 3s segments, each clip produces several predictions.
To score on the same 150 test CLIPS as every other model, we average a clip's
segment probabilities and take the argmax — "soft voting". This is also what the
API does at serving time for a segmented model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_proba_by_clip(clip_ids, proba, labels=None):
    """Average per-segment probabilities within each clip.

    Args:
        clip_ids: sequence of clip ids, one per segment row.
        proba:    (n_segments, n_classes) probability matrix.
        labels:   optional per-segment true label_idx (constant within a clip).

    Returns:
        (clip_order, clip_proba, clip_true) where clip_true is None if labels is
        None. clip_order is the list of unique clip ids in first-seen order.
    """
    df = pd.DataFrame({"clip_id": list(clip_ids)})
    proba = np.asarray(proba)
    clip_order = list(dict.fromkeys(df["clip_id"]))  # unique, order-preserving

    clip_proba = np.vstack([
        proba[(df["clip_id"] == cid).to_numpy()].mean(axis=0) for cid in clip_order
    ])

    clip_true = None
    if labels is not None:
        labels = np.asarray(labels)
        clip_true = np.array([
            labels[(df["clip_id"] == cid).to_numpy()][0] for cid in clip_order
        ])
    return clip_order, clip_proba, clip_true
