"""Model-agnostic predictor used by the serving API.

## What this code does
Wraps "audio in -> genre out" behind one interface so the FastAPI layer doesn't
care which model is loaded. Given a decoded waveform it:
1. **slides a window** over the audio (so a full 3-minute song is handled, not
   just a 30s clip), runs the right feature pipeline per window (librosa vector /
   PANNs embedding), and **averages** the per-window probabilities (soft voting);
2. returns the top genre, a confidence, the top-k guesses, and the full
   distribution — and flags the result **`uncertain`** when the top probability
   is below a configurable threshold (so out-of-distribution songs don't get a
   falsely confident single label).

The CNN path is single-shot (its mel-spectrogram step already fixes the length).

Config: `configs/serving.yaml` (`model.kind`, `inference.*`). The served model
can be overridden at runtime with the `MGC_SERVE_KIND` env var (the Docker image
sets it to `classic` to keep the container light).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from src.data.manifest import label_maps
from src.utils import PROJECT_ROOT, load_config


@dataclass
class Prediction:
    genre: str                       # top genre, or "uncertain" below threshold
    best_guess: str                  # always the argmax genre (even if uncertain)
    confidence: float                # probability of best_guess
    uncertain: bool
    top_k: list[dict]                # [{"genre": str, "prob": float}, ...]
    probabilities: dict[str, float]  # full distribution over all genres
    n_windows: int                   # how many windows were averaged


class Predictor:
    def __init__(self):
        scfg = load_config("serving")["model"]
        # Runtime override (e.g. Docker sets MGC_SERVE_KIND=classic).
        self.kind = os.environ.get("MGC_SERVE_KIND", scfg["kind"])

        inf = load_config("serving").get("inference", {})
        self.threshold = inf.get("confidence_threshold", 0.45)
        self.top_k = inf.get("top_k", 3)
        self.window_seconds = inf.get("window_seconds", 30)
        self.hop_seconds = inf.get("hop_seconds", 15)

        _, idx_to_genre = label_maps()
        self.class_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]
        self.sample_rate = load_config("data")["dataset"]["sample_rate"]
        self.fcfg = load_config("features")

        if self.kind == "classic":
            import joblib

            self.model = joblib.load(PROJECT_ROOT / scfg["classic_path"])
            self.name = scfg["name"]
        elif self.kind == "cnn":
            import torch

            from src.models.cnn import build_cnn

            arch = json.loads((PROJECT_ROOT / scfg["cnn_arch"]).read_text())
            self.model = build_cnn(arch)
            self.model.load_state_dict(
                torch.load(PROJECT_ROOT / scfg["cnn_weights"], map_location="cpu")
            )
            self.model.eval()
            self.name = "cnn"
        elif self.kind == "embeddings":
            import joblib

            self.model = joblib.load(PROJECT_ROOT / scfg["embeddings_path"])
            self.ecfg = load_config("embeddings")["panns"]
            self._tagger = None  # lazily loaded on first request
            # Surface which probe is loaded (e.g. embeddings_logreg_tuned) in /health.
            self.name = (PROJECT_ROOT / scfg["embeddings_path"]).stem
        else:
            raise ValueError(f"unknown model kind: {self.kind!r}")

    @property
    def description(self) -> str:
        return f"{self.kind}:{self.name}"

    # ---- windowing -----------------------------------------------------------
    def _windows(self, y: np.ndarray, sr: int) -> list[np.ndarray]:
        """Split a waveform into overlapping windows (or [y] if it's short)."""
        w = int(self.window_seconds * sr)
        h = max(1, int(self.hop_seconds * sr))
        if w <= 0 or len(y) <= w:
            return [y]
        starts = list(range(0, len(y) - w + 1, h))
        wins = [y[s : s + w] for s in starts]
        # capture a trailing window if a sizable tail is left uncovered
        if starts and (len(y) - (starts[-1] + w)) > 0.5 * w:
            wins.append(y[len(y) - w :])
        return wins

    # ---- prediction ----------------------------------------------------------
    def predict(self, y: np.ndarray) -> Prediction:
        """Decoded mono waveform (at self.sample_rate) -> Prediction."""
        if self.kind == "cnn":
            probs, n = self._predict_cnn(y), 1
        else:
            probs, n = self._predict_windowed(y)

        order = np.argsort(probs)[::-1]
        top = int(order[0])
        conf = float(probs[top])
        uncertain = conf < self.threshold
        return Prediction(
            genre="uncertain" if uncertain else self.class_names[top],
            best_guess=self.class_names[top],
            confidence=round(conf, 4),
            uncertain=uncertain,
            top_k=[{"genre": self.class_names[i], "prob": round(float(probs[i]), 4)}
                   for i in order[: self.top_k]],
            probabilities={g: round(float(p), 4) for g, p in zip(self.class_names, probs)},
            n_windows=n,
        )

    def _predict_windowed(self, y: np.ndarray) -> tuple[np.ndarray, int]:
        """Average per-window probabilities for classic / embeddings models."""
        if self.kind == "embeddings":
            y = self._to_panns_sr(y)
            sr = self.ecfg["sample_rate"]
            per_window = self._embed_probs
        else:  # classic
            sr = self.sample_rate
            per_window = self._classic_probs

        wins = self._windows(y, sr)
        probs = np.mean([per_window(w) for w in wins], axis=0)
        return probs, len(wins)

    def _classic_probs(self, y: np.ndarray) -> np.ndarray:
        from src.features.classic_features import extract_classic_features, feature_names

        names = feature_names(self.fcfg["classic"])
        feats = extract_classic_features(y, self.sample_rate, self.fcfg["classic"])
        vec = np.array([feats[n] for n in names]).reshape(1, -1)
        return self.model.predict_proba(vec)[0]

    def _to_panns_sr(self, y: np.ndarray) -> np.ndarray:
        if self.sample_rate != self.ecfg["sample_rate"]:
            import librosa

            y = librosa.resample(y, orig_sr=self.sample_rate, target_sr=self.ecfg["sample_rate"])
        return y

    def _embed_probs(self, y: np.ndarray) -> np.ndarray:
        from panns_inference import AudioTagging

        if self._tagger is None:  # load the heavy backbone once
            self._tagger = AudioTagging(
                checkpoint_path=str(PROJECT_ROOT / self.ecfg["checkpoint_path"]),
                device="cpu",
            )
        _, emb = self._tagger.inference(y[None, :])
        return self.model.predict_proba(np.asarray(emb).reshape(1, -1))[0]

    def _predict_cnn(self, y: np.ndarray) -> np.ndarray:
        import torch

        from src.features.melspec import compute_melspec

        spec = compute_melspec(y, self.sample_rate, self.fcfg["melspec"])
        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return torch.softmax(self.model(x), dim=1)[0].numpy()
