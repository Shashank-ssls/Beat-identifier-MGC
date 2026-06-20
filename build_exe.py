"""Build a standalone Windows executable for the Beat Identifier UI.

Wraps `run_ui.py` (which boots the FastAPI server + opens the browser) into a
self-contained app under `dist/BeatIdentifier/`. Bundles the light classic
**SVM** model only — no torch / PANNs — so the build stays small and reliable.

    python build_exe.py

The result is `dist/BeatIdentifier/BeatIdentifier.exe`; double-click it (or zip
the whole folder to share). To serve the heavier PANNs probe instead, run
`python run_ui.py` from the repo — that model is too large to bundle sensibly.
"""

from __future__ import annotations

import os

import PyInstaller.__main__

SEP = os.pathsep  # ";" on Windows, ":" elsewhere — used by --add-data


def data(src: str, dest: str) -> str:
    return f"--add-data={src}{SEP}{dest}"


ARGS = [
    "run_ui.py",
    "--name=BeatIdentifier",
    "--noconfirm",
    "--clean",
    "--console",  # keep a console so users can see the URL / close to stop
    # ── bundled data (resolved via sys._MEIPASS at runtime) ──
    data("configs", "configs"),
    data("models/classic_svm.joblib", "models"),
    data("src/api/static", "src/api/static"),
    # ── tricky deps: pull submodules + data + metadata wholesale ──
    "--collect-all=librosa",
    "--collect-all=sklearn",
    "--collect-all=soundfile",
    "--collect-all=soxr",
    "--collect-all=audioread",
    "--collect-all=pooch",
    "--collect-all=lazy_loader",
    "--collect-all=scipy",
    "--collect-submodules=numba",
    "--collect-submodules=llvmlite",
    # uvicorn/starlette/fastapi internals loaded dynamically
    "--collect-submodules=uvicorn",
    "--hidden-import=anyio",
    # ── exclude the heavy CNN/PANNs path ──
    # predictor.py imports these lazily inside the cnn/embeddings branches, which
    # the classic-SVM build never executes. Excluding them keeps the bundle from
    # ballooning to multi-GB (torch alone is ~3GB).
    "--exclude-module=torch",
    "--exclude-module=torchaudio",
    "--exclude-module=torchlibrosa",
    "--exclude-module=panns_inference",
    "--exclude-module=mlflow",
    "--exclude-module=xgboost",
    "--exclude-module=matplotlib",
    "--exclude-module=seaborn",
    "--exclude-module=pandas",
    "--exclude-module=sympy",
    "--exclude-module=IPython",
]


if __name__ == "__main__":
    PyInstaller.__main__.run(ARGS)
