"""Validate GTZAN audio files: catch corrupt/unreadable clips.

## What this code does
GTZAN ships with at least one broken file (`jazz.00054`) and, depending on the
copy you download, occasionally others. Before we trust the dataset we try to
actually decode each clip; anything that fails to load (or is the wrong length)
is reported so it can be excluded. The known-bad ids from `configs/data.yaml`
are treated as corrupt up front.

This module needs an audio backend (`librosa`/`soundfile`), so it only runs once
you've downloaded the data. The split manifest does NOT depend on it — that's
deliberate, so you can build a reproducible split before validating audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.utils import PROJECT_ROOT, load_config


@dataclass
class ValidationReport:
    """Result of scanning the audio directory."""

    ok: list[str] = field(default_factory=list)       # clip_ids that decoded fine
    corrupt: list[str] = field(default_factory=list)   # failed to decode / wrong length
    known_bad: list[str] = field(default_factory=list)  # excluded via config up front
    missing: list[str] = field(default_factory=list)   # expected but not found on disk

    def summary(self) -> str:
        return (
            f"ok={len(self.ok)} corrupt={len(self.corrupt)} "
            f"known_bad={len(self.known_bad)} missing={len(self.missing)}"
        )


def is_loadable(path: Path, sample_rate: int, min_seconds: float = 1.0) -> bool:
    """Return True if the clip decodes and is at least `min_seconds` long.

    librosa is imported lazily so this file can be imported in environments
    without the audio stack (the import only fails if you actually call this).
    """
    import librosa

    try:
        # sr=None keeps the file's native rate; we only care that it decodes.
        y, sr = librosa.load(path, sr=None, mono=True)
    except Exception:
        return False
    return len(y) >= int(min_seconds * sr)


def validate_dataset(audio_dir: Path | None = None) -> ValidationReport:
    """Decode every expected clip and bucket the results.

    Expected clips come from the canonical GTZAN layout in the config, minus the
    known-corrupt list (which goes straight into `known_bad`).
    """
    cfg = load_config("data")
    genres = cfg["dataset"]["genres"]
    sr = cfg["dataset"]["sample_rate"]
    corrupt_ids = set(cfg["corrupt_files"])
    if audio_dir is None:
        audio_dir = PROJECT_ROOT / cfg["dataset"]["audio_dir"]

    report = ValidationReport()
    for genre in genres:
        for i in range(100):  # canonical 100 clips per genre
            clip_id = f"{genre}.{i:05d}"
            if clip_id in corrupt_ids:
                report.known_bad.append(clip_id)
                continue

            path = audio_dir / genre / f"{clip_id}.wav"
            if not path.exists():
                report.missing.append(clip_id)
                continue

            if is_loadable(path, sample_rate=sr):
                report.ok.append(clip_id)
            else:
                report.corrupt.append(clip_id)

    return report


if __name__ == "__main__":
    rep = validate_dataset()
    print(rep.summary())
    if rep.corrupt:
        print("Newly-detected corrupt files (consider adding to configs/data.yaml):")
        for cid in rep.corrupt:
            print(f"  {cid}")
