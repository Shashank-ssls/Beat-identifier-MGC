"""Segmented classic-feature extraction: 3-second windows per clip.

## What this code does
The single biggest accuracy lever on GTZAN. Instead of one 83-feature vector per
30s clip, we slice each clip into non-overlapping 3s windows and extract the
SAME features from each window. That turns ~700 training clips into ~7000
training rows, which the classic models love.

Crucially, every segment keeps its parent clip's split label, so all segments of
a given clip stay together in train OR val OR test — no leakage. At evaluation we
average a clip's segment probabilities back into one clip-level prediction, so
the benchmark is still measured on the same 150 test CLIPS.

    python -m src.features.extract_segments            # full
    python -m src.features.extract_segments --limit 20 # smoke
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.data.manifest import load_clips
from src.features.classic_features import extract_classic_features, feature_names
from src.utils import PROJECT_ROOT, load_config


def segment_waveform(y: np.ndarray, seg_len: int) -> list[np.ndarray]:
    """Split into non-overlapping windows of seg_len samples (drop short tail)."""
    n = len(y) // seg_len
    return [y[i * seg_len : (i + 1) * seg_len] for i in range(n)]


def build_segmented_table(limit: int | None = None) -> "pd.DataFrame":
    import librosa

    dcfg = load_config("data")
    fcfg = load_config("features")
    sr = dcfg["dataset"]["sample_rate"]
    audio_dir = PROJECT_ROOT / dcfg["dataset"]["audio_dir"]
    seg_len = int(fcfg["classic"]["segment_seconds"] * sr)

    clips = load_clips()
    if limit:
        clips = clips[:limit]

    rows: list[dict] = []
    t0 = time.time()
    for n, clip in enumerate(clips, 1):
        y, _ = librosa.load(clip.path(audio_dir), sr=sr, mono=True)
        for seg_idx, seg in enumerate(segment_waveform(y, seg_len)):
            feats = extract_classic_features(seg, sr, fcfg["classic"])
            rows.append({
                "clip_id": clip.clip_id, "segment_idx": seg_idx,
                "genre": clip.genre, "label_idx": clip.label_idx,
                "split": clip.split, **feats,
            })
        if n % 50 == 0 or n == len(clips):
            print(f"  {n}/{len(clips)} clips  ({len(rows)} segments, {time.time() - t0:.0f}s)")

    cols = ["clip_id", "segment_idx", "genre", "label_idx", "split"] + feature_names(fcfg["classic"])
    return pd.DataFrame(rows)[cols]


def main() -> None:
    p = argparse.ArgumentParser(description="Extract 3s-segment classic features.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    df = build_segmented_table(limit=args.limit)
    out = PROJECT_ROOT / load_config("features")["classic"]["segment_cache_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote segmented table: {out}  shape={df.shape}")
    print(df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
