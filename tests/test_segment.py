"""Tests for 3s segmentation + clip-level aggregation (accuracy enhancement)."""

from __future__ import annotations

import numpy as np

from src.evaluation.aggregate import aggregate_proba_by_clip
from src.features.extract_segments import segment_waveform


def test_segment_waveform_count_and_length():
    sr, seg_seconds = 22050, 3
    seg_len = sr * seg_seconds
    y = np.zeros(30 * sr)  # 30s clip
    segs = segment_waveform(y, seg_len)
    assert len(segs) == 10              # 30s / 3s
    assert all(len(s) == seg_len for s in segs)


def test_segment_waveform_drops_short_tail():
    seg_len = 100
    y = np.zeros(350)  # 3 full segments + a 50-sample tail
    segs = segment_waveform(y, seg_len)
    assert len(segs) == 3               # tail dropped, no ragged segment


def test_aggregate_proba_averages_within_clip():
    # 2 clips, 2 segments each, 3 classes.
    clip_ids = ["a", "a", "b", "b"]
    proba = np.array([
        [0.6, 0.3, 0.1],   # a seg1 -> class0
        [0.2, 0.7, 0.1],   # a seg2 -> class1; avg = [0.4,0.5,0.1] -> class1
        [0.1, 0.1, 0.8],   # b seg1 -> class2
        [0.0, 0.2, 0.8],   # b seg2 -> class2; avg -> class2
    ])
    labels = np.array([1, 1, 2, 2])
    order, clip_proba, clip_true = aggregate_proba_by_clip(clip_ids, proba, labels)

    assert order == ["a", "b"]
    assert clip_proba.shape == (2, 3)
    np.testing.assert_allclose(clip_proba[0], [0.4, 0.5, 0.1])
    assert list(clip_proba.argmax(1)) == [1, 2]
    assert list(clip_true) == [1, 2]


def test_aggregate_without_labels():
    order, clip_proba, clip_true = aggregate_proba_by_clip(
        ["x", "x"], np.array([[0.9, 0.1], [0.7, 0.3]])
    )
    assert clip_true is None
    np.testing.assert_allclose(clip_proba[0], [0.8, 0.2])
