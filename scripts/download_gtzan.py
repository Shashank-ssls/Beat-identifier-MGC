"""Local fallback: download & extract GTZAN into data/.

## What this code does
On Kaggle you ATTACH the hosted GTZAN dataset (no download needed). This script
is the local-machine fallback so the repo is self-contained. It pulls the
dataset via `kagglehub` (the official, no-scraping route) and lays it out as
`data/genres_original/<genre>/<genre>.000NN.wav`, matching configs/data.yaml.

Requires Kaggle credentials configured for kagglehub (see README, Phase 1).
Run:  python scripts/download_gtzan.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Make `src` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import PROJECT_ROOT, load_config  # noqa: E402

KAGGLE_DATASET = "andradaolteanu/gtzan-dataset-music-genre-classification"


def main() -> None:
    cfg = load_config("data")
    dest = PROJECT_ROOT / cfg["dataset"]["audio_dir"]

    if dest.exists() and any(dest.iterdir()):
        print(f"Audio already present at {dest} — nothing to do.")
        return

    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "kagglehub is not installed. Install it with `pip install kagglehub`, "
            "or attach the GTZAN dataset directly if you are on Kaggle."
        )

    print(f"Downloading {KAGGLE_DATASET} via kagglehub ...")
    cached = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    # The Kaggle dataset nests the audio under a 'genres_original' folder.
    src_audio = next(cached.rglob("genres_original"), None)
    if src_audio is None:
        sys.exit(f"Could not find 'genres_original' under {cached}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying audio -> {dest}")
    shutil.copytree(src_audio, dest, dirs_exist_ok=True)
    print("Done. Now run: python -m src.data.split --scan")


if __name__ == "__main__":
    main()
