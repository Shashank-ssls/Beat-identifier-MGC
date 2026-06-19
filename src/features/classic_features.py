"""Path (A): hand-crafted librosa features for the classic-ML benchmark.

## What this code does
A from-scratch CNN learns its own features from the raw spectrogram. The classic
path instead hands the model a short, fixed-length vector of *musically
meaningful* numbers we compute ourselves. For each 30s clip we extract several
librosa features that each vary over time, then summarise each one by its **mean
and standard deviation** across the clip. Mean captures the typical value; std
captures how much it fluctuates. Concatenating them gives one feature vector per
clip — exactly what XGBoost / SVM expect.

What each feature means musically:
- MFCC: a compact description of timbre / "tone colour" (the workhorse of audio ML).
- chroma: how energy is distributed across the 12 pitch classes → harmony/key.
- spectral contrast: difference between peaks and valleys in the spectrum → how
  "tonal" vs "noisy" each frequency band is.
- zero-crossing rate: how often the waveform crosses zero → noisiness/percussiveness.
- spectral rolloff: the frequency below which most energy sits → brightness.
- tempo: estimated beats per minute → rhythm.
"""

from __future__ import annotations

import numpy as np


def feature_names(cfg: dict) -> list[str]:
    """Deterministic, ordered list of column names for the configured features.

    Kept in lock-step with `extract_classic_features` so the feature table's
    columns are stable and reproducible across runs/machines.
    """
    inc = cfg["include"]
    names: list[str] = []
    if inc["mfcc"]:
        for i in range(cfg["n_mfcc"]):
            names += [f"mfcc{i}_mean", f"mfcc{i}_std"]
    if inc["chroma"]:
        for i in range(12):  # 12 pitch classes
            names += [f"chroma{i}_mean", f"chroma{i}_std"]
    if inc["spectral_contrast"]:
        for i in range(7):  # librosa default: 6 bands + 1 → 7 rows
            names += [f"contrast{i}_mean", f"contrast{i}_std"]
    if inc["zero_crossing_rate"]:
        names += ["zcr_mean", "zcr_std"]
    if inc["spectral_rolloff"]:
        names += ["rolloff_mean", "rolloff_std"]
    if inc["tempo"]:
        names += ["tempo"]
    return names


def _mean_std(feats: dict, prefix: str, mat: np.ndarray) -> None:
    """Append per-row mean+std of a (n_rows, n_frames) feature matrix into feats."""
    for i in range(mat.shape[0]):
        feats[f"{prefix}{i}_mean"] = float(mat[i].mean())
        feats[f"{prefix}{i}_std"] = float(mat[i].std())


def extract_classic_features(y: np.ndarray, sr: int, cfg: dict) -> dict[str, float]:
    """Compute the configured features for one waveform → {name: value}.

    librosa is imported lazily so importing this module is cheap and doesn't
    require the audio stack until you actually extract features.
    """
    import librosa

    inc = cfg["include"]
    feats: dict[str, float] = {}

    if inc["mfcc"]:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg["n_mfcc"])
        _mean_std(feats, "mfcc", mfcc)
    if inc["chroma"]:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        _mean_std(feats, "chroma", chroma)
    if inc["spectral_contrast"]:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        _mean_std(feats, "contrast", contrast)
    if inc["zero_crossing_rate"]:
        zcr = librosa.feature.zero_crossing_rate(y)
        feats["zcr_mean"] = float(zcr.mean())
        feats["zcr_std"] = float(zcr.std())
    if inc["spectral_rolloff"]:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        feats["rolloff_mean"] = float(rolloff.mean())
        feats["rolloff_std"] = float(rolloff.std())
    if inc["tempo"]:
        # librosa.feature.tempo returns an array (one estimate here) → take [0].
        tempo = librosa.feature.tempo(y=y, sr=sr)
        feats["tempo"] = float(np.atleast_1d(tempo)[0])

    return feats
