"""Tests for feature extraction (Phase 2).

These verify the two contracts later phases depend on:
- classic features: the vector has exactly the configured length and is finite
  (no NaN/inf — XGBoost/SVM choke on those).
- mel-spectrograms: shape is (n_mels, frames), finite, float32, and every clip
  produces the SAME width (so the CNN can batch them).

They load a couple of real clips from the manifest, so they are skipped
automatically if the audio isn't present (e.g. on a clean CI checkout without
the dataset).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.manifest import load_clips
from src.features.classic_features import extract_classic_features, feature_names
from src.features.melspec import compute_melspec
from src.utils import PROJECT_ROOT, load_config

DCFG = load_config("data")
FCFG = load_config("features")
SR = DCFG["dataset"]["sample_rate"]
AUDIO_DIR = PROJECT_ROOT / DCFG["dataset"]["audio_dir"]


def _sample_clips(n=2):
    clips = load_clips()
    present = [c for c in clips if c.path(AUDIO_DIR).exists()]
    if len(present) < n:
        pytest.skip("GTZAN audio not present — skipping feature tests")
    return present[:n]


def _load(clip):
    import librosa

    y, _ = librosa.load(clip.path(AUDIO_DIR), sr=SR, mono=True)
    return y


def test_classic_vector_length_matches_names():
    clip = _sample_clips(1)[0]
    feats = extract_classic_features(_load(clip), SR, FCFG["classic"])
    names = feature_names(FCFG["classic"])
    assert set(feats) == set(names)
    assert len(feats) == len(names)


def test_classic_features_are_finite():
    for clip in _sample_clips(2):
        feats = extract_classic_features(_load(clip), SR, FCFG["classic"])
        values = np.array(list(feats.values()), dtype=float)
        assert np.isfinite(values).all(), "classic features contain NaN/inf"


def test_melspec_shape_and_dtype():
    clip = _sample_clips(1)[0]
    S = compute_melspec(_load(clip), SR, FCFG["melspec"])
    assert S.ndim == 2
    assert S.shape[0] == FCFG["melspec"]["n_mels"]
    assert S.dtype == np.float32
    assert np.isfinite(S).all(), "mel-spectrogram contains NaN/inf"


def test_melspec_fixed_width_across_clips():
    a, b = _sample_clips(2)
    Sa = compute_melspec(_load(a), SR, FCFG["melspec"])
    Sb = compute_melspec(_load(b), SR, FCFG["melspec"])
    # Fixed-length padding/truncation must give identical widths for batching.
    assert Sa.shape == Sb.shape


def test_melspec_is_normalised():
    clip = _sample_clips(1)[0]
    S = compute_melspec(_load(clip), SR, FCFG["melspec"])
    # z-score normalisation → roughly mean 0, std 1.
    assert abs(float(S.mean())) < 1e-3
    assert abs(float(S.std()) - 1.0) < 1e-2
