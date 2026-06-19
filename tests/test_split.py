"""Tests for the stratified split logic.

These are the guardrails for the project's anti-leakage promise: the split must
be reproducible, cover every clip exactly once, keep genre balance, and never
let a clip appear in two splits.
"""

from __future__ import annotations

from src.data.split import canonical_clip_ids, stratified_split
from src.utils import load_config

CFG = load_config("data")
GENRES = CFG["dataset"]["genres"]
CORRUPT = set(CFG["corrupt_files"])
SPLIT = CFG["split"]


def _split():
    clips = canonical_clip_ids(GENRES, CORRUPT)
    return clips, stratified_split(
        clips, SPLIT["train"], SPLIT["val"], SPLIT["test"], CFG["seed"]
    )


def test_corrupt_files_excluded():
    clips = canonical_clip_ids(GENRES, CORRUPT)
    clip_ids = {cid for _, cid in clips}
    for bad in CORRUPT:
        assert bad not in clip_ids


def test_every_clip_assigned_once():
    clips, rows = _split()
    assigned = [clip_id for _, clip_id, _ in rows]
    # No clip appears twice...
    assert len(assigned) == len(set(assigned))
    # ...and every discovered clip got a split.
    assert set(assigned) == {cid for _, cid in clips}


def test_splits_are_disjoint():
    _, rows = _split()
    buckets: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for _, clip_id, split in rows:
        buckets[split].add(clip_id)
    assert buckets["train"].isdisjoint(buckets["val"])
    assert buckets["train"].isdisjoint(buckets["test"])
    assert buckets["val"].isdisjoint(buckets["test"])


def test_stratified_per_genre_proportions():
    _, rows = _split()
    # Count train clips per genre; should be ~70% of that genre's clips.
    per_genre_total: dict[str, int] = {}
    per_genre_train: dict[str, int] = {}
    for genre, _, split in rows:
        per_genre_total[genre] = per_genre_total.get(genre, 0) + 1
        if split == "train":
            per_genre_train[genre] = per_genre_train.get(genre, 0) + 1
    for genre, total in per_genre_total.items():
        frac = per_genre_train[genre] / total
        assert abs(frac - SPLIT["train"]) < 0.02  # within 2 points


def test_reproducible_same_seed():
    clips = canonical_clip_ids(GENRES, CORRUPT)
    a = stratified_split(clips, SPLIT["train"], SPLIT["val"], SPLIT["test"], 42)
    b = stratified_split(clips, SPLIT["train"], SPLIT["val"], SPLIT["test"], 42)
    assert a == b


def test_different_seed_changes_assignment():
    clips = canonical_clip_ids(GENRES, CORRUPT)
    a = stratified_split(clips, SPLIT["train"], SPLIT["val"], SPLIT["test"], 42)
    b = stratified_split(clips, SPLIT["train"], SPLIT["val"], SPLIT["test"], 7)
    assert a != b
