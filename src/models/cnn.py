"""Small from-scratch CNN for genre classification + SpecAugment.

## What this code does
A deliberately small convolutional network that reads a mel-spectrogram "image"
and outputs a score for each of the 10 genres. It is kept small (4-5 conv
blocks) on purpose — that is the whole point of the benchmark, and it keeps the
model trainable on modest hardware (a 4GB GTX 1650).

### How a conv block works (plain English)
Each block does the same three things:
1. **Conv2d** — slides small learnable filters over the image to detect local
   patterns (edges, then textures, then genre-ish motifs in deeper blocks).
2. **BatchNorm + ReLU** — normalises activations (stabilises/speeds training)
   and keeps only positive signal (non-linearity).
3. **MaxPool 2x2** — halves height & width, keeping the strongest response →
   the network sees larger context with fewer numbers each block.
Channels double each block (more filters = richer features) while the spatial
size shrinks. At the end we average over what's left and a linear layer maps to
the 10 genre scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(
        self,
        conv_blocks: int = 4,
        base_channels: int = 16,
        num_classes: int = 10,
        dropout: float = 0.3,
        in_channels: int = 1,
    ):
        super().__init__()
        blocks = []
        in_ch = in_channels
        out_ch = base_channels
        for _ in range(conv_blocks):
            blocks.append(_conv_block(in_ch, out_ch))
            in_ch = out_ch
            out_ch *= 2  # double the filters each block
        self.features = nn.Sequential(*blocks)

        # Collapse whatever spatial size remains to 1x1 → fixed-size vector,
        # so the classifier works regardless of input width.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),  # in_ch = channels after last block
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class SpecAugment(nn.Module):
    """Light SpecAugment: zero out a random time band and a random freq band.

    This is cheap data augmentation done on the spectrogram itself. By randomly
    hiding a horizontal (frequency) and vertical (time) stripe each step, the
    network can't over-rely on one spot and generalises better. Applied during
    training only. (Implemented by hand so we don't need torchaudio.)
    """

    def __init__(self, time_mask_param: int = 30, freq_mask_param: int = 15):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, frames). Inputs are z-scored so 0 ≈ the mean.
        if not self.training:
            return x
        b, _, n_mels, n_frames = x.shape
        x = x.clone()
        for i in range(b):
            f = int(torch.randint(0, self.freq_mask_param + 1, (1,)))
            if f > 0 and n_mels > f:
                f0 = int(torch.randint(0, n_mels - f, (1,)))
                x[i, :, f0 : f0 + f, :] = 0.0
            t = int(torch.randint(0, self.time_mask_param + 1, (1,)))
            if t > 0 and n_frames > t:
                t0 = int(torch.randint(0, n_frames - t, (1,)))
                x[i, :, :, t0 : t0 + t] = 0.0
        return x


def build_cnn(model_cfg: dict) -> SmallCNN:
    """Construct a SmallCNN from the configs/cnn.yaml `model` section."""
    return SmallCNN(
        conv_blocks=model_cfg["conv_blocks"],
        base_channels=model_cfg["base_channels"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
    )
