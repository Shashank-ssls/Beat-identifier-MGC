"""Tests for the FastAPI serving app (Phase 6 + prediction-quality fixes).

Uses FastAPI's TestClient (no live server). Forces the light SVM model via
MGC_SERVE_KIND=classic so tests are fast and don't load the heavy PANNs backbone.
Skips automatically if the SVM model or a sample GTZAN clip isn't available.
"""

from __future__ import annotations

import io
import os

import pytest

# Force the light model BEFORE importing the app/predictor.
os.environ["MGC_SERVE_KIND"] = "classic"

from src.data.manifest import load_clips  # noqa: E402
from src.utils import PROJECT_ROOT, load_config  # noqa: E402

_classic_path = PROJECT_ROOT / load_config("serving")["model"]["classic_path"]
if not _classic_path.exists():
    pytest.skip("SVM model not trained — run train_classic", allow_module_level=True)


def _client():
    from fastapi.testclient import TestClient

    from src.api.app import app

    return TestClient(app)


def _sample_wav_bytes():
    dcfg = load_config("data")
    audio_dir = PROJECT_ROOT / dcfg["dataset"]["audio_dir"]
    clip = next((c for c in load_clips("test") if c.path(audio_dir).exists()), None)
    if clip is None:
        pytest.skip("GTZAN audio not present")
    return clip.path(audio_dir).read_bytes()


def test_health():
    with _client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"] == "classic:svm"


def test_predict_shape_and_fields():
    wav = _sample_wav_bytes()
    genres = sorted(load_config("data")["dataset"]["genres"])

    with _client() as client:
        r = client.post("/predict", files={"file": ("clip.wav", io.BytesIO(wav), "audio/wav")})
        assert r.status_code == 200
        b = r.json()

        # new fields from the prediction-quality fixes
        assert b["best_guess"] in genres
        assert b["genre"] == "uncertain" or b["genre"] in genres
        assert isinstance(b["uncertain"], bool)
        assert 0.0 <= b["confidence"] <= 1.0
        assert 1 <= len(b["top_k"]) <= 3
        assert b["top_k"][0]["genre"] == b["best_guess"]          # top_k is ranked
        assert b["n_windows"] >= 1
        # probabilities cover all genres and sum to ~1
        assert set(b["probabilities"]) == set(genres)
        assert abs(sum(b["probabilities"].values()) - 1.0) < 0.05


def test_uncertain_flag_consistency():
    wav = _sample_wav_bytes()
    with _client() as client:
        b = client.post("/predict", files={"file": ("c.wav", io.BytesIO(wav), "audio/wav")}).json()
    # genre is "uncertain" iff the uncertain flag is set
    assert (b["genre"] == "uncertain") == b["uncertain"]


def test_predict_rejects_empty_and_garbage():
    with _client() as client:
        assert client.post("/predict", files={"file": ("e.wav", io.BytesIO(b""), "audio/wav")}).status_code == 400
        assert client.post("/predict", files={"file": ("x.wav", io.BytesIO(b"nope"), "audio/wav")}).status_code == 400
