"""Path (A): hand-crafted librosa features for the classic-ML benchmark.

## What this code does
A from-scratch CNN learns its own features; the classic path instead hands the
model a short, fixed-length vector of *musically meaningful* numbers we compute
ourselves. For each clip we extract several librosa features that vary over time,
then summarise each by its **mean and standard deviation** (typical value +
fluctuation). Concatenating them gives one vector per clip for XGBoost / SVM.

What each feature captures:
- MFCC + their deltas: timbre / "tone colour" and how it changes over time.
- chroma: energy across the 12 pitch classes → harmony / key.
- spectral contrast: peak-vs-valley energy per band → tonal vs noisy.
- spectral centroid / bandwidth / rolloff: where the spectral energy sits and how
  spread/bright it is.
- spectral flatness: tone-like vs noise-like.
- zero-crossing rate: noisiness / percussiveness.
- RMS energy: loudness envelope.
- tempo: estimated beats per minute → rhythm.
"""

from __future__ import annotations

import numpy as np

N_CHROMA = 12
N_CONTRAST = 7  # librosa default: 6 bands + 1


def feature_names(cfg: dict) -> list[str]:
    """Deterministic, ordered column names for the configured features.

    Kept in lock-step with `extract_classic_features` so the feature table's
    columns are stable and reproducible. `include` keys default to True when
    absent, so older configs keep working.
    """
    inc = cfg["include"]

    def on(key: str) -> bool:
        return inc.get(key, True)

    names: list[str] = []
    if on("mfcc"):
        for i in range(cfg["n_mfcc"]):
            names += [f"mfcc{i}_mean", f"mfcc{i}_std"]
    if on("delta_mfcc"):
        for i in range(cfg["n_mfcc"]):
            names += [f"dmfcc{i}_mean", f"dmfcc{i}_std"]
    if on("chroma"):
        for i in range(N_CHROMA):
            names += [f"chroma{i}_mean", f"chroma{i}_std"]
    if on("spectral_contrast"):
        for i in range(N_CONTRAST):
            names += [f"contrast{i}_mean", f"contrast{i}_std"]
    for key, prefix in [
        ("spectral_centroid", "centroid"),
        ("spectral_bandwidth", "bandwidth"),
        ("spectral_flatness", "flatness"),
        ("zero_crossing_rate", "zcr"),
        ("spectral_rolloff", "rolloff"),
        ("rms", "rms"),
    ]:
        if on(key):
            names += [f"{prefix}_mean", f"{prefix}_std"]
    if on("tempo"):
        names += ["tempo"]
    return names


def _mean_std(feats: dict, prefix: str, mat: np.ndarray) -> None:
    """Append per-row mean+std of a (n_rows, n_frames) feature matrix."""
    for i in range(mat.shape[0]):
        feats[f"{prefix}{i}_mean"] = float(mat[i].mean())
        feats[f"{prefix}{i}_std"] = float(mat[i].std())


def _scalar(feats: dict, name: str, row: np.ndarray) -> None:
    feats[f"{name}_mean"] = float(row.mean())
    feats[f"{name}_std"] = float(row.std())


def extract_classic_features(y: np.ndarray, sr: int, cfg: dict) -> dict[str, float]:
    """Compute the configured features for one waveform -> {name: value}."""
    import librosa

    inc = cfg["include"]

    def on(key: str) -> bool:
        return inc.get(key, True)

    feats: dict[str, float] = {}

    if on("mfcc") or on("delta_mfcc"):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg["n_mfcc"])
        if on("mfcc"):
            _mean_std(feats, "mfcc", mfcc)
        if on("delta_mfcc"):
            _mean_std(feats, "dmfcc", librosa.feature.delta(mfcc))
    if on("chroma"):
        _mean_std(feats, "chroma", librosa.feature.chroma_stft(y=y, sr=sr))
    if on("spectral_contrast"):
        _mean_std(feats, "contrast", librosa.feature.spectral_contrast(y=y, sr=sr))
    if on("spectral_centroid"):
        _scalar(feats, "centroid", librosa.feature.spectral_centroid(y=y, sr=sr))
    if on("spectral_bandwidth"):
        _scalar(feats, "bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr))
    if on("spectral_flatness"):
        _scalar(feats, "flatness", librosa.feature.spectral_flatness(y=y))
    if on("zero_crossing_rate"):
        _scalar(feats, "zcr", librosa.feature.zero_crossing_rate(y))
    if on("spectral_rolloff"):
        _scalar(feats, "rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr))
    if on("rms"):
        _scalar(feats, "rms", librosa.feature.rms(y=y))
    if on("tempo"):
        tempo = librosa.feature.tempo(y=y, sr=sr)
        feats["tempo"] = float(np.atleast_1d(tempo)[0])

    return feats
