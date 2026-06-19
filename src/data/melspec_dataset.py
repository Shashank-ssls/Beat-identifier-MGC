"""PyTorch Dataset over the cached mel-spectrogram tensors.

## What this code does
Feeds the CNN. Each item is one clip's cached mel-spectrogram (the .npy written
in Phase 2) turned into a tensor of shape (1, n_mels, frames) — the leading 1 is
the single "image" channel a CNN expects — plus its integer genre label. Which
clips belong to train/val/test comes from the committed manifest, so the CNN
sees exactly the same split as the classic models.

torch is imported at module top here because this module is only ever used by
the CNN phases (which require torch anyway).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.manifest import load_clips
from src.utils import PROJECT_ROOT, load_config


class MelspecDataset(Dataset):
    """Loads cached (n_mels, frames) spectrograms for one split."""

    def __init__(self, split: str, limit: int | None = None):
        fcfg = load_config("features")
        self.cache_dir = PROJECT_ROOT / fcfg["melspec"]["cache_path"]

        clips = load_clips(split)
        # Keep only clips whose tensor was actually cached.
        self.clips = [c for c in clips if (self.cache_dir / f"{c.clip_id}.npy").exists()]
        if limit:
            self.clips = self.clips[:limit]
        if not self.clips:
            raise FileNotFoundError(
                f"No cached spectrograms for split={split!r} in {self.cache_dir}. "
                "Run `python -m src.features.extract --what melspec` first."
            )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip = self.clips[idx]
        spec = np.load(self.cache_dir / f"{clip.clip_id}.npy")  # (n_mels, frames)
        x = torch.from_numpy(spec).unsqueeze(0)  # add channel dim → (1, n_mels, frames)
        y = clip.label_idx
        return x, y

    @property
    def labels(self) -> list[int]:
        return [c.label_idx for c in self.clips]
