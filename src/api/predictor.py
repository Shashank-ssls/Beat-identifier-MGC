"""Model-agnostic predictor used by the serving API.

## What this code does
Wraps "audio in → genre out" behind one interface so the FastAPI layer doesn't
care which model is loaded. Given a decoded waveform it:
1. runs the right feature pipeline for the configured model (librosa feature
   vector for classic; mel-spectrogram for the CNN — the SAME code paths used in
   training, so serving and training can't drift), then
2. returns the predicted genre, a confidence (top probability), and the full
   probability distribution over all 10 genres.

The predictor is built once at app startup and reused for every request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src.data.manifest import label_maps
from src.utils import PROJECT_ROOT, load_config


@dataclass
class Prediction:
    genre: str
    confidence: float
    probabilities: dict[str, float]


class Predictor:
    def __init__(self):
        scfg = load_config("serving")["model"]
        self.kind = scfg["kind"]
        _, idx_to_genre = label_maps()
        self.class_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]

        dcfg = load_config("data")
        self.sample_rate = dcfg["dataset"]["sample_rate"]
        self.fcfg = load_config("features")

        # Optional: if set, the classic model was trained on N-second segments,
        # so at serving time we segment the upload and average segment probs.
        self.segment_seconds = scfg.get("segment_seconds")

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
        else:
            raise ValueError(f"unknown model kind: {self.kind!r}")

    @property
    def description(self) -> str:
        return f"{self.kind}:{self.name}"

    def predict(self, y: np.ndarray) -> Prediction:
        """Decoded mono waveform (at self.sample_rate) → Prediction."""
        if self.kind == "classic":
            probs = self._predict_classic(y)
        else:
            probs = self._predict_cnn(y)

        top = int(np.argmax(probs))
        return Prediction(
            genre=self.class_names[top],
            confidence=float(probs[top]),
            probabilities={g: float(p) for g, p in zip(self.class_names, probs)},
        )

    def _predict_classic(self, y: np.ndarray) -> np.ndarray:
        from src.features.classic_features import extract_classic_features, feature_names

        names = feature_names(self.fcfg["classic"])

        def vec_of(sig: np.ndarray) -> np.ndarray:
            feats = extract_classic_features(sig, self.sample_rate, self.fcfg["classic"])
            return np.array([feats[n] for n in names])

        if self.segment_seconds:
            # Segment the clip, predict each window, average (soft voting) —
            # mirrors how the segmented model was trained/evaluated.
            from src.features.extract_segments import segment_waveform

            seg_len = int(self.segment_seconds * self.sample_rate)
            segs = segment_waveform(y, seg_len) or [y]  # fall back to whole clip
            X = np.vstack([vec_of(s) for s in segs])
            return self.model.predict_proba(X).mean(axis=0)

        # Pipeline ends in a probability-capable estimator (SVC(probability=True)/XGB).
        return self.model.predict_proba(vec_of(y).reshape(1, -1))[0]

    def _predict_cnn(self, y: np.ndarray) -> np.ndarray:
        import torch

        from src.features.melspec import compute_melspec

        spec = compute_melspec(y, self.sample_rate, self.fcfg["melspec"])
        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,frames)
        with torch.no_grad():
            logits = self.model(x)
            return torch.softmax(logits, dim=1)[0].numpy()
