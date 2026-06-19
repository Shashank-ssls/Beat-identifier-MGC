# Music Genre Classification — Project Plan

**Setup:** Dual-approach benchmark (Classic ML vs. CNN) + full MLOps
**Hardware:** Kaggle GPU (training) + GTX 1650 / local (MLOps & serving)
**Level:** PyTorch beginner
**Dataset:** GTZAN (with known-flaw caveats handled)

---

## Why This Setup

Your answers picked the most ambitious combo (dual-approach benchmark + full MLOps) but you're a PyTorch beginner. That's a great way to learn, but the #1 resume risk is building something you can't explain. An interviewer won't ask about the classic ML side — they'll point at the CNN and the MLflow setup and ask "walk me through this." The plan is structured to be **learnable in stages**: build, understand, then move on.

**On the GTX 1650 (4GB VRAM):** This is the real constraint. Workable if disciplined:

- A from-scratch CNN on mel-spectrograms? Fine.
- Fine-tuning a large pretrained audio model? Will OOM. We avoid it.
- We use small batch sizes, mixed precision (AMP), and modest resolution.

**Dataset choice (GTZAN):** Not the best dataset (FMA is more credible), but at beginner level with 4GB VRAM, GTZAN's small size means you actually finish, iterate, and learn. The prompt adds a note about its label-leakage caveat so you can speak to it intelligently — knowing your dataset's flaws is a green flag in interviews. Swap to FMA-small if you want more credibility.

**Realistic expectation:** Classic ML (XGBoost on librosa features) often matches or beats a small from-scratch CNN on GTZAN. That's a genuinely interesting finding and a great talking point about when deep learning is and isn't worth it. The benchmark framing means you win either way.

---

## Where Each Part Runs (Kaggle + Local Split)

Kaggle's free tier gives you a P100/T4 with **16GB VRAM** (4x the 1650), ~30 GPU-hours/week, and GTZAN already hosted as an attachable dataset. For this project that's more than enough — the full benchmark trains in well under an hour of GPU time.

But Kaggle is notebook-first and **ephemeral**: sessions reset on idle/timeout and the filesystem doesn't persist unless you save outputs as datasets. That collides with the MLOps half (MLflow server, FastAPI, Docker, CI), which belongs in your repo, not a notebook.

So the clean division of labor:

| Concern | Runs on | Why |
|---|---|---|
| Feature extraction, CNN training, experiments | **Kaggle GPU** | 16GB VRAM, fast, GTZAN pre-hosted |
| MLflow tracking, FastAPI serving, Docker, CI | **Local** | Persistent, repo-based; serving barely uses GPU |
| Source of truth | **Git repo** | Kaggle is just a training runtime |

**Artifact flow:** train on Kaggle → log metrics/params to MLflow (point the notebook at a tracking URI, or export the run as files) → download the trained model weights as a Kaggle output → commit/register them locally → FastAPI loads from the local MLflow registry.

**Keep the AMP / small-batch discipline anyway.** Even though 16GB removes the OOM pressure, "I optimized for constrained 4GB hardware with mixed precision and small batches" is a real engineering talking point worth keeping in the code — it costs nothing and it's a skill you can defend in an interview.

---

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  1. DATA LAYER                                                    │
│     GTZAN (.wav, 30s clips, 10 genres)                            │
│     → validation (corrupt-file check, the known bad blues file)   │
│     → train/val/test split (stratified, fixed seed)               │
│     → split BEFORE feature extraction (prevent leakage)           │
└─────────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌──────────────────────┐         ┌──────────────────────────┐
│ 2A. CLASSIC ML PATH  │         │ 2B. DEEP LEARNING PATH    │
│  librosa features:   │         │  mel-spectrogram images   │
│  - MFCC (mean+std)   │         │  (128 mel bands)          │
│  - chroma            │         │  → normalized tensors     │
│  - spectral contrast │         │  → light augmentation     │
│  - zero-crossing,    │         │    (SpecAugment-style)    │
│    tempo, rolloff    │         │                           │
│  → feature vector    │         │  Small CNN (4-5 conv      │
│  → StandardScaler    │         │  blocks), AMP, batch=16   │
│  → XGBoost + SVM     │         │  → softmax over 10        │
└──────────────────────┘         └──────────────────────────┘
        │                                   │
        └─────────────────┬─────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. EXPERIMENT TRACKING (MLflow)                                  │
│     log: params, metrics (acc/F1/per-class), confusion matrix,    │
│          model artifacts, feature config → reproducible runs      │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. EVALUATION & COMPARISON                                       │
│     unified report: accuracy, macro-F1, per-genre confusion,      │
│     inference latency — classic vs CNN, on the SAME test set      │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. SERVING (FastAPI)                                             │
│     POST /predict (upload audio) → genre + confidence             │
│     loads best registered model from MLflow                       │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. PACKAGING & OPS                                               │
│     Dockerfile · docker-compose (api + mlflow) · GitHub Actions   │
│     CI (lint + tests) · README with results table & architecture  │
└─────────────────────────────────────────────────────────────────┘
```

### Repo structure the prompt will enforce

```
music-genre-classification/
├── data/                  # gitignored; scripts to download
├── src/
│   ├── data/              # validation, splitting, loaders
│   ├── features/          # librosa feature extraction
│   ├── models/            # classic.py, cnn.py
│   ├── training/          # train_classic.py, train_cnn.py
│   ├── evaluation/        # unified metrics + comparison
│   └── api/               # FastAPI app
├── configs/               # YAML configs (no hardcoded params)
├── tests/                 # pytest
├── notebooks/             # 01_eda, 02_results_analysis
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

---

## The Claude Code Prompt

Two critical things about this prompt. First, it instructs Claude Code to build in **phases and stop for your review** between each — this keeps it learnable instead of a 3000-line dump you can't defend. Second, it bakes in your **hardware constraints** so it doesn't generate code that OOMs your 1650.

Copy everything in the block below into Claude Code:

````markdown
# PROJECT: Music Genre Classification — Benchmark + MLOps

You are helping me (a PyTorch beginner) build a resume-quality music genre
classification system.

## RUNTIME SPLIT (important)
- GPU-heavy work (feature extraction, CNN training, experiments) runs on
  KAGGLE (free P100/T4, 16GB VRAM, GTZAN pre-hosted as a Kaggle dataset).
  Provide these as runnable notebook cells / scripts I can paste into Kaggle.
- MLOps (MLflow server, FastAPI serving, Docker, CI) runs LOCALLY on my
  machine (GTX 1650, 4GB). The Git repo is the source of truth; Kaggle is
  just a training runtime.
- For each phase, tell me clearly WHERE it runs (Kaggle vs local) and how
  artifacts (trained weights, MLflow runs) move from Kaggle back to the repo.

## CRITICAL WORKING AGREEMENT
- Build this in PHASES. After each phase, STOP, summarize what you built,
  and wait for me to confirm before continuing. Do NOT build everything at once.
- I am a beginner. In every phase, add a short "## What this code does"
  explanation in plain English, and add inline comments on any non-obvious
  audio-DSP or PyTorch line. I must be able to explain every file in an interview.
- Prefer clarity over cleverness. No premature abstraction.
- All hyperparameters live in YAML config files in `configs/`. Never hardcode.
- Use a fixed random seed (42) everywhere for reproducibility.

## HARDWARE / TRAINING CONSTRAINTS
- Training happens on Kaggle (16GB VRAM), so OOM is not a hard limit — but
  KEEP good discipline anyway as a deliberate engineering choice: use mixed
  precision (torch.cuda.amp), keep the CNN small (4-5 conv blocks), and keep
  batch size modest (32 is fine on Kaggle; the code should still run at <=16
  so it also works on my local 4GB 1650 for sanity checks). Add a comment
  noting this dual-target design.
- Do NOT fine-tune large pretrained audio models — keep the CNN from-scratch
  and small; that is the point of the benchmark.
- Mel-spectrograms at 128 mel bands. Cache extracted features to disk / as a
  Kaggle output dataset so I don't recompute every run.

## DATASET
- GTZAN (10 genres, 1000 clips, 30s, .wav). It is already available as a
  Kaggle dataset — attach it to the notebook rather than downloading. Still
  provide a local download script as a fallback for the repo.
- IMPORTANT: GTZAN has a known corrupt file (jazz.00054) and documented
  label-leakage concerns. Handle the corrupt file and add a README note
  explaining the leakage caveat and why we split BEFORE feature extraction.
- Stratified train/val/test split with a fixed seed. Split first, then extract
  features, to prevent leakage. Save the split manifest so it is reproducible
  across both Kaggle and local runs.

## GOAL
Benchmark TWO approaches on the SAME test set:
  (A) Classic ML: librosa features (MFCC, chroma, spectral contrast, ZCR,
      rolloff, tempo) → StandardScaler → XGBoost and SVM.
  (B) Deep learning: small CNN on mel-spectrograms with light SpecAugment-style
      augmentation and AMP.
Track everything in MLflow. Serve the best model via FastAPI. Containerize.
Add CI.

## TECH STACK
Python 3.10+, librosa, scikit-learn, xgboost, PyTorch, MLflow, FastAPI,
pytest, Docker, GitHub Actions. Pin versions in requirements.txt.

## PHASES (stop after each)

### Phase 0 — Scaffolding [LOCAL]
Create the repo structure, requirements.txt, .gitignore, configs/ skeleton,
and a README outline. Explain the structure.

### Phase 1 — Data layer [KAGGLE + repo]
Attach GTZAN on Kaggle. Corrupt-file handling, validation, stratified split
(saved to disk as a manifest committed to the repo so splits are reproducible).
A small EDA notebook (class balance, sample waveforms/spectrograms).

### Phase 2 — Feature extraction [KAGGLE]
(A) librosa feature pipeline → cached feature table (saved as a Kaggle output).
(B) mel-spectrogram generation → cached tensors (saved as a Kaggle output).
Write pytest tests verifying output shapes and that no NaNs appear.

### Phase 3 — Classic ML [KAGGLE, then sync to local MLflow]
Train XGBoost + SVM, log metrics/params/confusion-matrix. Show how to export
the runs/weights from Kaggle so I can register them in local MLflow. Explain
what the features represent musically.

### Phase 4 — CNN [KAGGLE]
Small CNN + Dataset/DataLoader, AMP, early stopping, modest batch. Log metrics.
Export trained weights as a Kaggle output dataset I can download into the repo.
Explain each layer in plain English.

### Phase 5 — Unified evaluation [LOCAL]
One script that evaluates ALL models on the SAME held-out test set and emits a
comparison table (accuracy, macro-F1, per-class F1, inference latency) plus a
results notebook. This is the centerpiece of the resume story.

### Phase 6 — Serving [LOCAL]
FastAPI app: POST /predict accepts an audio file, returns genre + confidence.
Loads the best model from the local MLflow registry. Include a /health endpoint.

### Phase 7 — Packaging & CI [LOCAL]
Dockerfile, docker-compose (api + mlflow), GitHub Actions running lint + pytest.
Finalize README with: architecture diagram, results table, the Kaggle/local
runtime split, how-to-run, and the GTZAN leakage caveat written so I can speak
to it.

Begin with Phase 0 only. Stop and wait for my confirmation.
````

---

## How to Use This Well

The phased "stop and wait" structure is the most important line in there. Let it finish a phase, then actually **read** the code and ask for an explanation of anything fuzzy before you say "continue." That's the difference between a project you built and a project you can defend.
