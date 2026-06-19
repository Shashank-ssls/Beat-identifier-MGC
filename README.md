# Music Genre Classification — Benchmark + MLOps

Classify 30-second audio clips into 10 genres, **benchmarking two approaches on
the same held-out test set**:

- **(A) Classic ML** — `librosa` audio features → `StandardScaler` → XGBoost & SVM
- **(B) Deep learning** — a small from-scratch CNN on mel-spectrograms (AMP, light SpecAugment)

…wrapped in a full MLOps loop: MLflow experiment tracking, a FastAPI prediction
service, Docker packaging, and GitHub Actions CI.

> **Why a benchmark?** On GTZAN, classic ML on hand-crafted features often
> matches or beats a small from-scratch CNN. The benchmark framing makes that a
> *finding* about when deep learning is worth it — not a failure.

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

## Repo structure

```
music-genre-classification/
├── data/                  # gitignored; raw audio + cached features
├── src/
│   ├── data/              # validation, splitting, loaders        (Phase 1)
│   ├── features/          # librosa features + mel-spectrograms   (Phase 2)
│   ├── models/            # classic.py, cnn.py                    (Phase 3-4)
│   ├── training/          # train_classic.py, train_cnn.py        (Phase 3-4)
│   ├── evaluation/        # unified metrics + comparison          (Phase 5)
│   └── api/               # FastAPI app                           (Phase 6)
├── configs/               # YAML configs — no hardcoded params
├── tests/                 # pytest
├── notebooks/             # 01_eda, 02_results_analysis
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
| `configs/features.yaml` | classic feature set + mel-spectrogram params |
| `configs/classic.yaml` | XGBoost & SVM hyperparameters |
| `configs/cnn.yaml` | CNN architecture, training, augmentation |

---

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
```

**PyTorch on a local GTX 1650 (4GB):** `requirements.txt` pins the default
`torch`/`torchaudio` build. To use your local CUDA GPU, install the matching
CUDA build from <https://pytorch.org/get-started/locally/> instead (e.g. the
`cu121` wheels). On Kaggle a CUDA build is already preinstalled — skip the local
torch install there.

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
- [ ] **Phase 3 — Classic ML** *(Kaggle → local MLflow)* — XGBoost + SVM
- [ ] **Phase 4 — CNN** *(Kaggle)* — small CNN, AMP, early stopping
- [ ] **Phase 5 — Unified evaluation** *(local)* — same-test-set comparison table + notebook
- [ ] **Phase 6 — Serving** *(local)* — FastAPI `/predict` + `/health`
- [ ] **Phase 7 — Packaging & CI** *(local)* — Docker, docker-compose, GitHub Actions

---

## Results

_Populated in Phase 5 — comparison of accuracy, macro-F1, per-class F1, and
inference latency for XGBoost vs SVM vs CNN on the same test set._
