"""Tests for the FastAPI serving app (Phase 6).

Uses FastAPI's TestClient (no live server). Skips automatically if the trained
model or a sample GTZAN clip isn't available, so the suite stays green on a
clean checkout.
"""

from __future__ import annotations

import io

import pytest

from src.data.manifest import load_clips
from src.utils import PROJECT_ROOT, load_config

# Skip the whole module if the served model isn't trained yet.
_scfg = load_config("serving")["model"]
_model_path = PROJECT_ROOT / _scfg["classic_path"]
if _scfg["kind"] == "classic" and not _model_path.exists():
    pytest.skip("served model not trained — run train_classic", allow_module_level=True)


def _client():
    from fastapi.testclient import TestClient

    from src.api.app import app

    # `with` triggers the lifespan handler that loads the model.
    return TestClient(app)


def _sample_wav_bytes():
    dcfg = load_config("data")
    audio_dir = PROJECT_ROOT / dcfg["dataset"]["audio_dir"]
    clip = next((c for c in load_clips("test") if c.path(audio_dir).exists()), None)
    if clip is None:
        pytest.skip("GTZAN audio not present")
    return clip.path(audio_dir).read_bytes(), clip.genre


def test_health():
    with _client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"]  # e.g. "classic:svm"


def test_predict_returns_valid_genre():
    wav, _true_genre = _sample_wav_bytes()
    dcfg = load_config("data")
    genres = sorted(dcfg["dataset"]["genres"])

    with _client() as client:
        r = client.post("/predict", files={"file": ("clip.wav", io.BytesIO(wav), "audio/wav")})
        assert r.status_code == 200
        body = r.json()
        assert body["genre"] in genres
        assert 0.0 <= body["confidence"] <= 1.0
        # probabilities cover all genres and (roughly) sum to 1
        assert set(body["probabilities"]) == set(genres)
        assert abs(sum(body["probabilities"].values()) - 1.0) < 0.05


def test_predict_rejects_empty_file():
    with _client() as client:
        r = client.post("/predict", files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")})
        assert r.status_code == 400


def test_predict_rejects_garbage_audio():
    with _client() as client:
        r = client.post("/predict", files={"file": ("x.wav", io.BytesIO(b"not audio"), "audio/wav")})
        assert r.status_code == 400
