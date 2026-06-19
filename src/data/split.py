"""Build the reproducible stratified train/val/test split manifest.

## What this code does
This is the single most important "anti-leakage" step in the project. GTZAN has
documented label leakage (near-duplicate excerpts), so we decide *which clip
goes into which split BEFORE extracting any features* and freeze that decision
into a committed CSV manifest (`data/split_manifest.csv`). Every later phase —
on Kaggle or locally — reads this manifest, so a clip can never leak from train
into test just because someone re-ran feature extraction.

The split is **stratified**: each genre is split 70/15/15 independently, so all
three splits keep the same genre balance.

How clips are discovered:
- By default we generate the *canonical* GTZAN file list (10 genres x 100 clips,
  named `<genre>.00000` ... `<genre>.00099`) and drop the known-corrupt files.
  This means the manifest is reproducible even before you've downloaded the
  audio — handy for committing it to the repo.
- `--scan` instead walks an actual audio directory, so it adapts if you use a
  subset or a differently-laid-out copy.

This module deliberately uses only the Python standard library so it runs in any
environment without the full ML stack installed.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from src.utils import PROJECT_ROOT, load_config

# Canonical GTZAN layout: 100 clips per genre, indices 00000..00099.
CLIPS_PER_GENRE = 100
AUDIO_EXTS = (".wav", ".au")


def canonical_clip_ids(genres: list[str], corrupt: set[str]) -> list[tuple[str, str]]:
    """Return (genre, clip_id) pairs for the standard GTZAN layout.

    clip_id looks like 'blues.00000'. Known-corrupt ids are skipped.
    """
    clips: list[tuple[str, str]] = []
    for genre in genres:
        for i in range(CLIPS_PER_GENRE):
            clip_id = f"{genre}.{i:05d}"
            if clip_id in corrupt:
                continue
            clips.append((genre, clip_id))
    return clips


def scan_clip_ids(audio_dir: Path, genres: list[str], corrupt: set[str]) -> list[tuple[str, str]]:
    """Walk an actual GTZAN directory (audio_dir/<genre>/*.wav) for clips."""
    clips: list[tuple[str, str]] = []
    for genre in genres:
        genre_dir = audio_dir / genre
        if not genre_dir.is_dir():
            raise FileNotFoundError(f"Expected genre folder missing: {genre_dir}")
        for path in sorted(genre_dir.iterdir()):
            if path.suffix.lower() not in AUDIO_EXTS:
                continue
            clip_id = path.stem  # e.g. 'blues.00000'
            if clip_id in corrupt:
                continue
            clips.append((genre, clip_id))
    return clips


def stratified_split(
    clips: list[tuple[str, str]],
    train: float,
    val: float,
    test: float,
    seed: int,
) -> list[tuple[str, str, str]]:
    """Split clips per-genre into train/val/test.

    Returns (genre, clip_id, split) triples. Each genre is shuffled with the
    seeded RNG and partitioned independently so genre balance is preserved in
    every split. The test set takes the remainder so the three fractions always
    account for 100% of clips (no rounding gaps).
    """
    assert abs((train + val + test) - 1.0) < 1e-6, "split fractions must sum to 1.0"

    rng = random.Random(seed)

    # Group clip ids by genre.
    by_genre: dict[str, list[str]] = {}
    for genre, clip_id in clips:
        by_genre.setdefault(genre, []).append(clip_id)

    rows: list[tuple[str, str, str]] = []
    for genre in sorted(by_genre):
        ids = sorted(by_genre[genre])  # sort first → deterministic before shuffle
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        # test gets whatever is left → fractions always cover every clip
        splits = (
            [(genre, cid, "train") for cid in ids[:n_train]]
            + [(genre, cid, "val") for cid in ids[n_train : n_train + n_val]]
            + [(genre, cid, "test") for cid in ids[n_train + n_val :]]
        )
        rows.extend(splits)
    return rows


def write_manifest(rows: list[tuple[str, str, str]], out_path: Path) -> None:
    """Write the split manifest CSV: columns genre, clip_id, split, label_idx."""
    # label_idx = the genre's integer class index (alphabetical genre order).
    genres_sorted = sorted({genre for genre, _, _ in rows})
    label_idx = {g: i for i, g in enumerate(genres_sorted)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["genre", "clip_id", "split", "label_idx"])
        for genre, clip_id, split in sorted(rows):
            writer.writerow([genre, clip_id, split, label_idx[genre]])


def build_manifest(scan: bool = False) -> Path:
    """End-to-end: read config, discover clips, split, write the manifest CSV."""
    cfg = load_config("data")
    genres = cfg["dataset"]["genres"]
    corrupt = set(cfg["corrupt_files"])
    split_cfg = cfg["split"]
    seed = cfg["seed"]

    if scan:
        audio_dir = PROJECT_ROOT / cfg["dataset"]["audio_dir"]
        clips = scan_clip_ids(audio_dir, genres, corrupt)
    else:
        clips = canonical_clip_ids(genres, corrupt)

    rows = stratified_split(
        clips,
        train=split_cfg["train"],
        val=split_cfg["val"],
        test=split_cfg["test"],
        seed=seed,
    )

    out_path = PROJECT_ROOT / split_cfg["manifest_path"]
    write_manifest(rows, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GTZAN split manifest.")
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan the real audio directory instead of using the canonical GTZAN file list.",
    )
    args = parser.parse_args()

    out_path = build_manifest(scan=args.scan)

    # Tiny summary so you can eyeball the result.
    counts: dict[str, int] = {}
    with out_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            counts[row["split"]] = counts.get(row["split"], 0) + 1
    total = sum(counts.values())
    print(f"Wrote {total} clips to {out_path}")
    for split in ("train", "val", "test"):
        n = counts.get(split, 0)
        print(f"  {split:5s}: {n:4d}  ({n / total:.1%})")


if __name__ == "__main__":
    main()
