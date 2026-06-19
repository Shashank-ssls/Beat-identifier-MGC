"""Read the committed split manifest and resolve clip file paths.

## What this code does
Every later phase asks the same question: "give me the list of clips in the
train (or val/test) split, with their labels and file paths." This module is the
single place that answers it, reading `data/split_manifest.csv` produced by
`split.py`. Keeping this in one place means classic-ML and CNN code can never
disagree about which clip is in which split.

Uses only the standard library so it imports cleanly anywhere.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.utils import PROJECT_ROOT, load_config


@dataclass(frozen=True)
class Clip:
    """One audio clip from the manifest."""

    clip_id: str       # e.g. 'blues.00000'
    genre: str         # e.g. 'blues'
    label_idx: int     # integer class index (alphabetical genre order)
    split: str         # 'train' | 'val' | 'test'

    def path(self, audio_dir: Path) -> Path:
        """Resolve the .wav path under a given GTZAN audio root."""
        return audio_dir / self.genre / f"{self.clip_id}.wav"


def manifest_path() -> Path:
    cfg = load_config("data")
    return PROJECT_ROOT / cfg["split"]["manifest_path"]


def load_clips(split: str | None = None) -> list[Clip]:
    """Load clips from the manifest, optionally filtered to one split.

    >>> train = load_clips("train")
    >>> all_clips = load_clips()        # every split
    """
    path = manifest_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {path}. "
            "Run `python -m src.data.split` to generate it."
        )

    clips: list[Clip] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if split is not None and row["split"] != split:
                continue
            clips.append(
                Clip(
                    clip_id=row["clip_id"],
                    genre=row["genre"],
                    label_idx=int(row["label_idx"]),
                    split=row["split"],
                )
            )
    return clips


def label_maps() -> tuple[dict[str, int], dict[int, str]]:
    """Return (genre -> idx, idx -> genre) using the config's genre order."""
    cfg = load_config("data")
    genres = sorted(cfg["dataset"]["genres"])
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    idx_to_genre = {i: g for g, i in genre_to_idx.items()}
    return genre_to_idx, idx_to_genre
