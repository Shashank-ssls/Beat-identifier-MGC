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
| ⭐ **PANNs CNN14 embeddings + tuned probe** | **0.880** | **0.879** |
| PANNs embeddings + linear probe | 0.873 | 0.873 |
| Ensemble (SVM + PANNs, soft-vote) | 0.867 | 0.867 |
| XGBoost (Optuna-tuned) | 0.820 | 0.821 |
| SVM (RBF) on librosa features | 0.780 | 0.782 |
| CNN (from scratch) | 0.693 | 0.668 |

**Headline takeaways**
- 📈 **Transfer learning wins** — pretrained audio embeddings beat both hand-crafted features and a from-scratch CNN on a small dataset.
- 🎛️ **Tuning pays off where it can** — Optuna lifts the probe to **0.880** and XGBoost from 0.74 → 0.82; the soft-vote ensemble lands between the probe and the classics.
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

## Run it once (end to end)

```bash
# 0. Environment
python -m venv .venv
.venv\Scripts\activate                 # Unix: source .venv/bin/activate
pip install -r requirements.txt
# For an NVIDIA GPU, install the CUDA torch build instead of the default CPU one:
#   pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 1. Data — get GTZAN into data/genres_original/<genre>/*.wav
python scripts/download_gtzan.py       # needs a Kaggle API token; or copy it in manually
python -m src.data.split               # reproducible split (manifest already committed)

# 2. Features (cached to features_cache/)
python -m src.features.extract                 # librosa table + mel-spectrograms
python -m src.features.extract_embeddings      # PANNs CNN14 embeddings (downloads checkpoint*)

# 3. Train the models  (each logs to ./mlruns and saves to ./models)
python -m src.training.train_classic           # SVM + XGBoost (librosa features)
python -m src.training.train_cnn --device auto  # from-scratch CNN (GPU if available)
python -m src.training.train_embeddings        # PANNs linear probe  ← best

# 4. (optional) accuracy extras
python -m src.features.extract_segments && python -m src.training.train_classic_segmented
python -m src.training.tune --target probe --trials 30      # Optuna-tune the probe
python -m src.training.tune --target xgboost --trials 40    # Optuna-tune XGBoost

# 5. Evaluate everything on the same test set → reports/comparison.csv
python -m src.evaluation.evaluate

# 6. Serve it
uvicorn src.api.app:app --reload       # → http://127.0.0.1:8000/docs  (upload a clip)
```

\* **PANNs checkpoint (Windows):** `panns_inference` downloads via `wget`, which
Windows lacks. Fetch it manually once into `models/panns_cnn14.pth`:
```bash
curl -L -o models/panns_cnn14.pth "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
```
(and drop `class_labels_indices.csv` into `~/panns_data/` — see the script header).

> **Minimum path to a working demo:** steps 0, 1, `train_classic`, then step 6 —
> that serves the SVM with no GPU and no PANNs download.

---

## Web UI & desktop app

The API serves a minimal browser UI at the root (`http://127.0.0.1:8000/`): a
single card with a drag-and-drop file picker on top and the predicted genre +
top-3 probability bars below. No build step — it's a static page served by
FastAPI.

For a one-click, no-terminal experience there's a Windows launcher and a
PyInstaller build:

```bash
python run_ui.py        # boots the server and opens the UI in your browser
python build_exe.py     # → dist/MusicGenreClassifier/MusicGenreClassifier.exe
```

The packaged `.exe` bundles the light **classic SVM** only (no torch/PANNs), so
it stays ~400 MB and starts in seconds — double-click it, or zip the folder to
share. To serve the more accurate PANNs probe, run `run_ui.py` from the repo with
`MGC_SERVE_KIND=embeddings` (too large to bundle sensibly).

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
| **PANNs CNN14 embeddings + Optuna-tuned probe** | **0.880** | **0.879** | 0.19* |
| PANNs CNN14 embeddings + linear probe | 0.873 | 0.873 | 0.20* |
| Ensemble — SVM + PANNs probe (soft-vote) | 0.867 | 0.867 | — |
| XGBoost (Optuna-tuned, 30s) | 0.820 | 0.821 | 1.40 |
| SVM — 3s segments + tuned (GroupKFold) | 0.800 | 0.802 | 7.24 |
| SVM (RBF), 30s | 0.780 | 0.782 | 0.23 |
| XGBoost (30s) | 0.740 | 0.741 | 1.00 |
| CNN (from-scratch, GTX 1650) | 0.693 | 0.668 | 18.4 |

\* The probe itself is sub-ms, but the PANNs model serves only *after* a CNN14
forward pass to produce the embedding (~tens of ms on CPU) — so its real
end-to-end latency is much higher than the SVM's. Accuracy vs serving-cost is the
trade-off.

**Finding 1 — transfer learning wins; from-scratch CNN loses.** A linear probe
on frozen **PANNs CNN14** embeddings (pretrained on AudioSet) tops the board at
**0.880** once Optuna tunes its regularisation (**0.873** untuned) — the only
thing that breaks past the hand-feature ceiling. Meanwhile the *from-scratch*
CNN (0.693) loses to even the hand-feature SVM (0.780): with ~700 training clips
a small CNN can't out-learn hand features, but a model pretrained on millions of
clips can. That contrast is the core lesson — *when you have little data, borrow
features from a big pretrained model.*

**Finding 2 — tuning helps the classics, an ensemble doesn't beat the best single
model.** Optuna lifts XGBoost from 0.74 → **0.82** (richer librosa features +
searched depth/learning-rate), closing most of the gap to the embeddings. But a
soft-vote **ensemble** of the SVM and the PANNs probe (0.867) lands *below* the
probe alone — the SVM's errors aren't independent enough to add signal, so
averaging just pulls the strong model toward the weak one.

**Finding 3 — "3-second segmentation" gives no real gain when done without
leakage.** Splitting each clip into 3s windows (~10x rows) + soft-voting is the
classic trick behind the ~90% GTZAN figures online — but those use a *random*
segment split that leaks segments of the same recording across train/test. With
a **track-grouped** split + GroupKFold tuning, the honest gain is ~0 (0.780 →
0.800): voting over segments ≈ averaging features over the full clip. The
inflated 90% is largely a leakage artifact.

Full per-class F1 table: `reports/comparison.csv`; analysis in
`notebooks/03_results_analysis.ipynb`.

Reproduce: `python -m src.training.train_classic`,
`python -m src.features.extract_segments && python -m src.training.train_classic_segmented`,
`python -m src.features.extract_embeddings && python -m src.training.train_embeddings`,
`python -m src.training.train_cnn --device cuda`, then tune the tabular models with
`python -m src.training.tune --target xgboost --trials 40` and `--target probe --trials 30`
(all log to `./mlruns`; view with `mlflow ui --backend-store-uri ./mlruns`).

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
