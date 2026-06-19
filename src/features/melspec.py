"""Path (B): mel-spectrogram tensors for the CNN.

## What this code does
The CNN doesn't get hand-crafted numbers — it gets an "image" of the sound and
learns features itself. That image is a **mel-spectrogram**: time on the x-axis,
mel-scaled frequency on the y-axis, and colour = energy. We:
1. Pad/truncate every clip to a fixed number of samples so all spectrograms have
   the SAME width (a CNN batches fixed-size tensors).
2. Compute the mel-spectrogram with the params from configs/features.yaml.
3. Convert to a dB (log) scale — loudness is perceived logarithmically, and this
   makes quiet detail visible to the network.
4. Z-score normalise (mean 0, std 1) so inputs are on a stable scale for training.

Output: a float32 array of shape (n_mels, n_frames), cached as one .npy per clip.
"""

from __future__ import annotations

import numpy as np


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    """Truncate or zero-pad a waveform to exactly `target_len` samples."""
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)))


def compute_melspec(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Waveform → normalised log-mel-spectrogram of shape (n_mels, n_frames)."""
    import librosa

    # 1. fixed length → fixed-width spectrogram across all clips
    y = fix_length(y, int(cfg["duration_seconds"] * sr))

    # 2. mel-spectrogram (power). fmax=None means Nyquist (sr/2).
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=cfg["n_mels"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        fmin=cfg["fmin"],
        fmax=cfg["fmax"],
    )

    # 3. power → dB (log scale), referenced to the clip's peak
    if cfg.get("to_db", True):
        S = librosa.power_to_db(S, ref=np.max)

    # 4. per-spectrogram z-score (eps guards against a silent/constant clip)
    if cfg.get("normalize", True):
        S = (S - S.mean()) / (S.std() + 1e-8)

    return S.astype(np.float32)
