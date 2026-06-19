# Serving image for the Music Genre Classifier API (default = SVM model).
# Small on purpose: no torch/mlflow — just the classic serving stack.
FROM python:3.12-slim

# libsndfile1 = soundfile/librosa audio decoding; ffmpeg = broader format support.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached across code changes.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code + configs (the trained model is mounted at runtime, see compose).
COPY src/ src/
COPY configs/ configs/

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
