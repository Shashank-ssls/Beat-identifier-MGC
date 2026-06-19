# 🎵 Music Genre Classification — Benchmark + MLOps

[![CI](https://github.com/Shashank-ssls/Beat-identifier-MGC/actions/workflows/ci.yml/badge.svg)](https://github.com/Shashank-ssls/Beat-identifier-MGC/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Lint: ruff](https://img.shields.io/badge/lint-ruff-orange)
![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)

An end-to-end, reproducible system that classifies 30-second audio clips into 10
genres and **benchmarks three modeling approaches on the same held-out test
set** — wrapped in a production-style MLOps loop (experiment tracking, a REST
API, containerization, and CI).

**Approaches compared**

| | Approach | How |
|---|---|---|
| **A** | Classic ML | `librosa` features → `StandardScaler` → XGBoost & SVM |
| **B** | Deep learning | small **from-scratch CNN** on mel-spectrograms (AMP, SpecAugment) |
| **C** | Transfer learning | linear probe on frozen **PANNs CNN14** embeddings ⭐ |

### Results at a glance — held-out test set (150 clips), ranked by macro-F1

| Model | Accuracy | Macro-F1 |
|---|---|---|
| ⭐ **PANNs CNN14 embeddings + linear probe** | **0.873** | **0.873** |
| SVM (RBF) on librosa features | 0.807 | 0.807 |
| XGBoost | 0.713 | 0.714 |
| CNN (from scratch) | 0.693 | 0.668 |

**Headline takeaways**
- 📈 **Transfer learning wins** — pretrained audio embeddings beat both hand-crafted features and a from-scratch CNN on a small dataset.
- 🧪 **The from-scratch CNN loses to a plain SVM** — with only ~700 training clips, "deep learning" is not automatically better.
- 🕵️ **Honest evaluation** — track-grouped splits show the famous "~90% on GTZAN" results are largely a **data-leakage artifact**.

> Built as a hands-on portfolio project to practice the full ML lifecycle end to
> end — raw audio → reproducible training → experiment tracking → a deployed API.

---

## Where things run (Kaggle + local split)

| Concern | Runs on | Why |
|---|---|---|
| Feature extraction, CNN training, experiments | **Kaggle GPU** | Free 16GB P100/T4, GTZAN pre-hosted |
| MLflow tracking, FastAPI serving, Docker, CI | **Local** | Persistent, repo-based; serving barely uses GPU |
| Source of truth | **Git repo** | Kaggle is just a training runtime |

**Artifact flow:** train on Kaggle → log metrics/params to MLflow → download
trained weights as a Kaggle output → register them in local MLflow → FastAPI
loads the best model from the local registry.

---

## Architecture

```
            GTZAN .wav (10 genres x 100 clips)
                          │
        ┌─────────────────┴─────────────────┐
        │  Data layer (split BEFORE features)│  stratified 70/15/15,
        │  validate · corrupt-file drop      │  committed manifest (anti-leak)
        └─────────────────┬─────────────────┘
        ┌──────────┬───────┴───────┬───────────────┐
        ▼          ▼               ▼               ▼
   librosa     mel-spec        3s segments     PANNs CNN14
   features    (128 bands)     (10x rows)      embeddings (2048-d)
        │          │               │               │
        ▼          ▼               ▼               ▼
   XGBoost/SVM  from-scratch    tuned SVM       linear probe
                CNN (AMP)       (GroupKFold)    (transfer learn)
        └──────────┴───────┬───────┴───────────────┘
                           ▼
                MLflow  (params · metrics · confusion matrices · models)
                           ▼
            Unified eval  → reports/comparison.csv (same test set)
                           ▼
            FastAPI  /predict  /health   →   Docker · docker-compose · CI
```

---

## Quickstart

```bash
python -m venv .venv && . .venv/Scripts/activate   # (Unix: source .venv/bin/activate)
pip install -r requirements.txt

python -m src.data.split                      # reproducible split (already committed)
python -m src.features.extract                # librosa features + mel-spectrograms
python -m src.training.train_classic          # XGBoost + SVM  → MLflow + models/
python -m src.training.train_cnn --device auto # CNN (uses GPU if available)
python -m src.evaluation.evaluate             # comparison table → reports/
uvicorn src.api.app:app --reload              # serve → http://127.0.0.1:8000/docs
```

Optional accuracy enhancements (see Results):
```bash
python -m src.features.extract_segments    && python -m src.training.train_classic_segmented
python -m src.features.extract_embeddings  && python -m src.training.train_embeddings   # best, 0.873
```

---

## Repo structure

```
Beat-identifier-MGC/
├── data/                  # gitignored; raw audio + cached features
├── src/
│   ├── data/              # validation, splitting, loaders        (Phase 1)
│   ├── features/          # librosa features + mel-spectrograms   (Phase 2)
│   ├── models/            # classic.py, cnn.py                    (Phase 3-4)
│   ├── training/          # train_classic.py, train_cnn.py        (Phase 3-4)
│   ├── evaluation/        # unified metrics + comparison          (Phase 5)
│   └── api/               # FastAPI app                           (Phase 6)
├── configs/               # YAML configs — no hardcoded params
├── tests/                 # pytest (33 tests)
├── notebooks/             # 01_eda, 02_kaggle_train_cnn, 03_results_analysis
├── Dockerfile             # (Phase 7)
├── docker-compose.yml     # (Phase 7)
├── .github/workflows/     # CI: lint + tests                      (Phase 7)
├── requirements.txt
└── README.md
```

---

## Configuration

All hyperparameters live in `configs/` — nothing is hardcoded, and the random
seed (`42`) is fixed everywhere for reproducibility.

| File | Controls |
|---|---|
| `configs/data.yaml` | dataset paths, genres, corrupt-file list, train/val/test split |
| `configs/features.yaml` | classic feature set, mel-spectrogram + segmentation params |
| `configs/classic.yaml` | XGBoost & SVM hyperparameters + SVM sweep grid |
| `configs/cnn.yaml` | CNN architecture, training, augmentation |
| `configs/embeddings.yaml` | PANNs backbone + linear-probe settings |
| `configs/serving.yaml` | which model the API serves |

---

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
```

**PyTorch / GPU:** plain `pip install torch` on Windows gives a **CPU-only**
build. To use an NVIDIA GPU (this project was trained on a 4GB GTX 1650),
install the matching CUDA build from
<https://pytorch.org/get-started/locally/> — e.g.
`pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121`.
Training the small CNN fits in 4GB with mixed precision; Kaggle's free 16GB GPU
is an optional alternative (see `notebooks/02_kaggle_train_cnn.ipynb`).

---

## Dataset — GTZAN (and its caveats)

GTZAN: 10 genres × 100 clips × 30s `.wav`. On Kaggle, **attach** the hosted
dataset rather than downloading; a local download script is provided as a repo
fallback (Phase 1).

Two documented flaws this project handles explicitly:

1. **Corrupt file** — `jazz.00054` has a broken header and is dropped during
   validation (see `configs/data.yaml → corrupt_files`).
2. **Label leakage** — GTZAN contains repeated/near-duplicate excerpts from the
   same recordings across splits, which can inflate accuracy. We mitigate by
   doing the **stratified split BEFORE feature extraction** and saving a
   reproducible split manifest, so the same clip never spans train and test.
   *(Full leakage discussion expanded in Phase 7.)*

---

## Build phases

This project is built in reviewable phases. Status:

- [x] **Phase 0 — Scaffolding** *(local)* — repo structure, configs, requirements, README outline
- [x] **Phase 1 — Data layer** *(Kaggle + repo)* — validation, stratified split, EDA notebook
- [x] **Phase 2 — Feature extraction** *(Kaggle/local)* — librosa feature table + cached mel-spectrograms, shape/NaN tests
- [x] **Phase 3 — Classic ML** *(local + MLflow)* — XGBoost + SVM, tracked in MLflow
- [x] **Phase 4 — CNN** *(local CPU smoke + Kaggle GPU)* — small CNN, SpecAugment, AMP, early stopping
- [x] **Phase 5 — Unified evaluation** *(local)* — same-test-set comparison table + notebook
- [x] **Phase 6 — Serving** *(local)* — FastAPI `/predict` + `/health`, loads the best model
- [x] **Phase 7 — Packaging & CI** *(local)* — Dockerfile, docker-compose (api + mlflow), GitHub Actions

---

## Serving the model

The FastAPI app loads the best model (SVM by default — see `configs/serving.yaml`)
once at startup and exposes:

| Endpoint | Description |
|---|---|
| `GET /health` | liveness + which model is loaded (e.g. `classic:svm`) |
| `POST /predict` | upload an audio file → `{genre, confidence, probabilities}` |

```bash
uvicorn src.api.app:app --reload          # then open http://127.0.0.1:8000/docs
curl -F "file=@some_clip.wav" http://127.0.0.1:8000/predict
```

To serve a different model, set `model.kind` in `configs/serving.yaml` to `cnn`
or `embeddings` (PANNs — most accurate but a heavier container).

### Docker

```bash
docker compose up --build
#   api    → http://localhost:8000/docs
#   mlflow → http://localhost:5000
```

The image is intentionally small — it installs only the SVM serving stack
(`requirements-serve.txt`, no torch). Your trained model in `./models` is
mounted read-only at runtime (models are gitignored, not baked into the image).

### CI

`.github/workflows/ci.yml` runs on every push/PR: **ruff** lint, **pytest**
(data/model-dependent tests auto-skip when GTZAN/weights aren't present), and a
**Docker build** to validate the image. No secrets required.

---

## Results

Held-out **test set** (150 clips), ranked by macro-F1:

| Model | Test accuracy | Test macro-F1 | Classifier latency (ms/clip) |
|---|---|---|---|
| **PANNs CNN14 embeddings + linear probe** | **0.873** | **0.873** | 0.27* |
| SVM — 3s segments + tuned (GroupKFold) | 0.813 | 0.813 | 6.5 |
| SVM (RBF), 30s | 0.807 | 0.807 | 0.24 |
| XGBoost | 0.713 | 0.714 | 1.03 |
| CNN (from-scratch, GTX 1650) | 0.693 | 0.668 | 22.1 |

\* The probe itself is sub-ms, but the PANNs model serves only *after* a CNN14
forward pass to produce the embedding (~tens of ms on CPU) — so its real
end-to-end latency is much higher than the SVM's. Accuracy vs serving-cost is the
trade-off.

**Finding 1 — transfer learning wins; from-scratch CNN loses.** A linear probe
on frozen **PANNs CNN14** embeddings (pretrained on AudioSet) tops the board at
**0.873** — the only thing that breaks past the hand-feature ceiling. Meanwhile
the *from-scratch* CNN (0.693) loses to even the hand-feature SVM (0.807):
with ~700 training clips a small CNN can't out-learn hand features, but a model
pretrained on millions of clips can. That contrast is the core lesson — *when
you have little data, borrow features from a big pretrained model.*

**Finding 2 — "3-second segmentation" gives no real gain when done without
leakage.** Splitting each clip into 3s windows (~10x rows) + soft-voting is the
classic trick behind the ~90% GTZAN figures online — but those use a *random*
segment split that leaks segments of the same recording across train/test. With
a **track-grouped** split + GroupKFold tuning, the honest gain is ~0 (0.807 →
0.813): voting over segments ≈ averaging features over the full clip. The
inflated 90% is largely a leakage artifact.

Full per-class F1 table: `reports/comparison.csv`; analysis in
`notebooks/03_results_analysis.ipynb`.

Reproduce: `python -m src.training.train_classic`,
`python -m src.features.extract_segments && python -m src.training.train_classic_segmented`,
`python -m src.features.extract_embeddings && python -m src.training.train_embeddings`,
and `python -m src.training.train_cnn --device cuda` (all log to `./mlruns`; view
with `mlflow ui --backend-store-uri ./mlruns`).

---

## ⚠️ Disclaimer

This is an **educational / portfolio project**, not a production service.

- **Closed-set classifier.** It only knows the 10 GTZAN genres (blues, classical,
  country, disco, hiphop, jazz, metal, pop, reggae, rock). Given audio outside
  these — or any non-music input — it returns the *nearest* of the 10, never
  "unknown". Expect low / split confidence on out-of-distribution songs.
- **Trained on GTZAN**, a small (~1000-clip, early-2000s) research dataset with
  documented quality and licensing caveats. The models reflect that dataset's
  biases and are **not** robust real-world genre taggers.
- **No commercial use.** The GTZAN dataset and the pretrained PANNs CNN14 weights
  carry their own terms — respect them. Results (~0.87 macro-F1) are reported
  under a leakage-safe, track-grouped split and are intentionally lower than the
  inflated numbers commonly quoted for GTZAN.

---

## 📄 License

Released under the [MIT License](LICENSE) © 2026 Shashank Singhal.

The MIT license covers the **original code in this repository only**. The GTZAN
dataset and PANNs CNN14 pretrained weights are the property of their respective
authors and are used here for research / educational purposes under their own
terms.

---

## 🙏 Acknowledgements

- **GTZAN** — G. Tzanetakis & P. Cook, *Musical Genre Classification of Audio
  Signals* (2002).
- **PANNs** — Kong et al., *PANNs: Large-Scale Pretrained Audio Neural Networks
  for Audio Pattern Recognition* (2020).
- Built with PyTorch, scikit-learn, XGBoost, librosa, MLflow, and FastAPI.
