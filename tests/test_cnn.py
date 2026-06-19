"""Tests for the CNN path (Phase 4): model + SpecAugment shapes.

CPU-only and dataset-free — tiny synthetic tensors keep CI fast and don't need
GTZAN or a GPU. torch is the only heavy import; the whole CNN phase needs it.
"""

from __future__ import annotations

import torch

from src.models.cnn import SpecAugment, build_cnn
from src.utils import load_config

CFG = load_config("cnn")


def _dummy_batch(b=4, n_mels=128, frames=256):
    return torch.randn(b, 1, n_mels, frames)


def test_cnn_output_shape():
    model = build_cnn(CFG["model"]).eval()
    out = model(_dummy_batch())
    # one score per genre for each item in the batch
    assert out.shape == (4, CFG["model"]["num_classes"])


def test_cnn_handles_variable_width():
    # AdaptiveAvgPool means different input widths still produce the same logits shape.
    model = build_cnn(CFG["model"]).eval()
    assert model(_dummy_batch(frames=200)).shape == model(_dummy_batch(frames=400)).shape


def test_specaugment_preserves_shape_and_is_noop_in_eval():
    aug = SpecAugment(
        time_mask_param=CFG["augmentation"]["time_mask_param"],
        freq_mask_param=CFG["augmentation"]["freq_mask_param"],
    )
    x = _dummy_batch()

    aug.eval()
    assert torch.equal(aug(x), x)  # no masking when not training

    aug.train()
    out = aug(x)
    assert out.shape == x.shape


def test_cnn_backward_runs():
    model = build_cnn(CFG["model"]).train()
    x = _dummy_batch()
    y = torch.randint(0, CFG["model"]["num_classes"], (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    # at least one parameter received a gradient
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
