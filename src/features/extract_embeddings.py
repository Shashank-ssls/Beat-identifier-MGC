"""Extract PANNs CNN14 embeddings (pretrained audio features) per clip.

## What this code does
Instead of hand-crafted features OR a from-scratch CNN, this uses a CNN14 model
**pretrained on Google AudioSet** (2M+ YouTube clips) as a frozen feature
extractor. For each GTZAN clip it produces one 2048-dim embedding that already
encodes rich, general audio structure. A simple linear classifier on top
("linear probe") then beats everything else — this is transfer learning.

The pretrained checkpoint (~340 MB) downloads once to `models/` (on F:, not C:).
Embeddings are cached so we extract them only once.

    python -m src.features.extract_embeddings            # full
    python -m src.features.extract_embeddings --limit 20 # smoke
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.data.manifest import load_clips
from src.utils import PROJECT_ROOT, load_config


def build_embeddings_table(limit: int | None = None) -> "pd.DataFrame":
    import librosa
    import torch
    from panns_inference import AudioTagging

    dcfg = load_config("data")
    ecfg = load_config("embeddings")["panns"]
    sr = ecfg["sample_rate"]
    audio_dir = PROJECT_ROOT / dcfg["dataset"]["audio_dir"]
    ckpt = PROJECT_ROOT / ecfg["checkpoint_path"]
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading PANNs CNN14 on {device} (checkpoint: {ckpt})")
    tagger = AudioTagging(checkpoint_path=str(ckpt), device=device)

    clips = load_clips()
    if limit:
        clips = clips[:limit]

    rows: list[dict] = []
    dim = ecfg["embedding_dim"]
    t0 = time.time()
    for n, clip in enumerate(clips, 1):
        y, _ = librosa.load(clip.path(audio_dir), sr=sr, mono=True)
        # PANNs expects a batch dimension: (batch, samples).
        _, emb = tagger.inference(y[None, :])
        emb = np.asarray(emb).reshape(-1)[:dim]
        row = {"clip_id": clip.clip_id, "genre": clip.genre,
               "label_idx": clip.label_idx, "split": clip.split}
        row.update({f"emb_{i}": float(v) for i, v in enumerate(emb)})
        rows.append(row)
        if n % 50 == 0 or n == len(clips):
            print(f"  {n}/{len(clips)} clips  ({time.time() - t0:.0f}s)")

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract PANNs CNN14 embeddings.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    df = build_embeddings_table(limit=args.limit)
    out = PROJECT_ROOT / load_config("embeddings")["panns"]["cache_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote embeddings table: {out}  shape={df.shape}")


if __name__ == "__main__":
    main()
