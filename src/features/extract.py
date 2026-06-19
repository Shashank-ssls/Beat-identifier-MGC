"""Build the cached feature artifacts for BOTH benchmark paths.

## What this code does
Walks the committed split manifest, loads each clip's audio exactly once, and
feeds it to both feature pipelines:
  (A) classic → one row in a feature table  → saved as a single parquet file
  (B) melspec → one (128, frames) tensor     → saved as one .npy per clip
Caching means later phases (Kaggle training, local eval) never recompute
features — they just load these artifacts.

Where it runs: locally (your extracted data) or on Kaggle (point audio_dir at
the attached dataset and save the outputs as a Kaggle output dataset).

Examples:
    python -m src.features.extract --limit 20       # quick smoke run
    python -m src.features.extract --what melspec    # only the CNN tensors
    python -m src.features.extract                    # full classic + melspec
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.manifest import Clip, load_clips
from src.features.classic_features import extract_classic_features, feature_names
from src.features.melspec import compute_melspec
from src.utils import PROJECT_ROOT, load_config


def _load_audio(clip: Clip, audio_dir: Path, sr: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(clip.path(audio_dir), sr=sr, mono=True)
    return y


def extract_all(limit: int | None = None, what: str = "both") -> None:
    dcfg = load_config("data")
    fcfg = load_config("features")
    sr = dcfg["dataset"]["sample_rate"]
    audio_dir = PROJECT_ROOT / dcfg["dataset"]["audio_dir"]

    do_classic = what in ("classic", "both")
    do_melspec = what in ("melspec", "both")

    clips = load_clips()
    if limit:
        clips = clips[:limit]

    mel_dir = PROJECT_ROOT / fcfg["melspec"]["cache_path"]
    if do_melspec:
        mel_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t0 = time.time()
    for n, clip in enumerate(clips, 1):
        y = _load_audio(clip, audio_dir, sr)

        if do_classic:
            feats = extract_classic_features(y, sr, fcfg["classic"])
            rows.append(
                {
                    "clip_id": clip.clip_id,
                    "genre": clip.genre,
                    "label_idx": clip.label_idx,
                    "split": clip.split,
                    **feats,
                }
            )

        if do_melspec:
            S = compute_melspec(y, sr, fcfg["melspec"])
            np.save(mel_dir / f"{clip.clip_id}.npy", S)

        if n % 50 == 0 or n == len(clips):
            print(f"  {n}/{len(clips)} clips  ({time.time() - t0:.0f}s)")

    if do_classic:
        # Stable column order: metadata first, then the configured feature names.
        cols = ["clip_id", "genre", "label_idx", "split"] + feature_names(fcfg["classic"])
        df = pd.DataFrame(rows)[cols]
        out = PROJECT_ROOT / fcfg["classic"]["cache_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"Wrote classic feature table: {out}  shape={df.shape}")

    if do_melspec:
        print(f"Wrote {len(clips)} mel-spectrograms to {mel_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Extract & cache features for both paths.")
    p.add_argument("--limit", type=int, default=None, help="only process the first N clips")
    p.add_argument("--what", choices=["classic", "melspec", "both"], default="both")
    args = p.parse_args()
    extract_all(limit=args.limit, what=args.what)


if __name__ == "__main__":
    main()
