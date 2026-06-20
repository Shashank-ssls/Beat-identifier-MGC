"""Fine-tune the PANNs CNN14 backbone end-to-end (vs. the frozen linear probe).

## What this code does
The linear probe (`train_embeddings`) freezes PANNs and only trains a classifier
on its embeddings. This script instead **fine-tunes the backbone itself** on
GTZAN: it loads pretrained CNN14, swaps in a fresh 10-class head, and trains
(optionally unfreezing the conv stack). Fine-tuning usually beats the frozen
probe by a few points — at much higher compute cost.

## ⚠️ Runs on a big GPU, NOT a 4GB card
CNN14 (~80M params) over 30s of 32kHz audio needs well more than 4GB once you add
gradients + optimizer state. **Run this on Kaggle's 16GB GPU** (see
`notebooks/02_kaggle_train_cnn.ipynb` for the clone/attach pattern), or any
>=8GB GPU. It is intentionally NOT part of the local test suite.

    python -m src.training.finetune_panns --epochs 20 --batch-size 8
    python -m src.training.finetune_panns --freeze-backbone   # head-only (lighter)
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.manifest import label_maps, load_clips
from src.evaluation.metrics import compute_metrics
from src.utils import PROJECT_ROOT, load_config, set_seed

MODELS_DIR = PROJECT_ROOT / "models"


class RawAudioDataset(Dataset):
    """Loads raw mono audio at PANNs' sample rate, padded/truncated to fixed len."""

    def __init__(self, split: str, sr: int, seconds: int = 30):
        dcfg = load_config("data")
        self.audio_dir = PROJECT_ROOT / dcfg["dataset"]["audio_dir"]
        self.sr, self.n = sr, sr * seconds
        self.clips = [c for c in load_clips(split) if c.path(self.audio_dir).exists()]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        import librosa

        c = self.clips[i]
        y, _ = librosa.load(c.path(self.audio_dir), sr=self.sr, mono=True)
        y = y[: self.n] if len(y) >= self.n else np.pad(y, (0, self.n - len(y)))
        return torch.from_numpy(y).float(), c.label_idx


class PannsClassifier(nn.Module):
    """Pretrained CNN14 backbone -> embedding -> fresh linear head (10 classes)."""

    def __init__(self, ckpt: str, sr: int, num_classes: int = 10, freeze: bool = False):
        super().__init__()
        from panns_inference.models import Cnn14

        # AudioSet config CNN14 was pretrained with.
        self.backbone = Cnn14(sample_rate=sr, window_size=1024, hop_size=320,
                              mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        state = torch.load(ckpt, map_location="cpu")["model"]
        self.backbone.load_state_dict(state)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Linear(2048, num_classes)

    def forward(self, x):
        emb = self.backbone(x)["embedding"]   # (B, 2048)
        return self.head(emb)


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune PANNs CNN14 on GTZAN.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU detected — this will be extremely slow. Use Kaggle.")

    ecfg = load_config("embeddings")["panns"]
    sr = ecfg["sample_rate"]
    ckpt = str(PROJECT_ROOT / ecfg["checkpoint_path"])
    _, idx_to_genre = label_maps()
    class_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]

    loaders = {
        s: DataLoader(RawAudioDataset(s, sr), batch_size=args.batch_size,
                      shuffle=(s == "train"), num_workers=args.num_workers)
        for s in ("train", "val", "test")
    }

    model = PannsClassifier(ckpt, sr, freeze=args.freeze_backbone).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        yt, yp = [], []
        for x, y in loaders[split]:
            x = x.to(device)
            logits = model(x)
            yp.append(logits.argmax(1).cpu().numpy())
            yt.append(y.numpy())
        return compute_metrics(np.concatenate(yt), np.concatenate(yp), class_names)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        vm = evaluate("val")
        print(f"  epoch {epoch:3d}  val_acc={vm['accuracy']:.3f}  val_macroF1={vm['macro_f1']:.3f}")
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    tm = evaluate("test")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "panns_finetuned.pt")
    print(f"\n  PANNs fine-tuned  test_acc={tm['accuracy']:.3f}  test_macroF1={tm['macro_f1']:.3f}")
    print(f"  saved -> {MODELS_DIR / 'panns_finetuned.pt'}")


if __name__ == "__main__":
    main()
