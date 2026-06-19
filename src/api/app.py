"""FastAPI serving app: predict a music genre from an uploaded audio clip.

## What this code does
A tiny web service around the trained model:
- `GET  /health`  → liveness + which model is loaded (for uptime checks / k8s).
- `POST /predict` → upload an audio file (multipart), get back the predicted
  genre, a confidence score, and the full probability distribution.

The model is loaded ONCE at startup (via the lifespan handler) and reused for
every request, so we don't pay model-load cost per call.

Run locally:
    uvicorn src.api.app:app --reload
Then open http://127.0.0.1:8000/docs for an interactive upload form.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

import librosa
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.api.predictor import Predictor
from src.utils import load_config

# Holds the singleton predictor; populated at startup.
state: dict[str, Predictor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor()  # load model once
    yield
    state.clear()


scfg = load_config("serving")
app = FastAPI(title=scfg["api"]["title"], lifespan=lifespan)
MAX_BYTES = scfg["api"]["max_upload_mb"] * 1024 * 1024


@app.get("/health")
def health():
    predictor = state.get("predictor")
    return {
        "status": "ok" if predictor else "loading",
        "model": predictor.description if predictor else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="file too large")

    predictor: Predictor = state["predictor"]
    try:
        # Decode the uploaded bytes to a mono waveform at the model's sample rate.
        y, _ = librosa.load(io.BytesIO(content), sr=predictor.sample_rate, mono=True)
    except Exception as exc:  # unreadable / unsupported audio
        raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}")

    if y.size == 0:
        raise HTTPException(status_code=400, detail="decoded audio is empty")

    result = predictor.predict(y)
    return {
        "filename": file.filename,
        "model": predictor.description,
        "genre": result.genre,
        "confidence": round(result.confidence, 4),
        "probabilities": {g: round(p, 4) for g, p in result.probabilities.items()},
    }
