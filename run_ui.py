"""Desktop launcher: boot the FastAPI server and open the UI in the browser.

This is the entry point bundled into the Windows executable (see build_exe.py).
Run it directly with `python run_ui.py`, or double-click the built .exe.

Model selection (most accurate by default):

    python run_ui.py              # PANNs CNN14 probe (0.880) — the best model
    python run_ui.py --classic    # light classic SVM (0.78) — instant, no torch

The packaged .exe always uses the classic SVM (it ships without torch/PANNs to
stay small), regardless of flags.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

HOST = os.environ.get("MGC_HOST", "127.0.0.1")
PORT = int(os.environ.get("MGC_PORT", "8000"))

FROZEN = getattr(sys, "frozen", False)


def _resolve_kind(args: argparse.Namespace) -> str:
    """Pick which model to serve. The bundled exe has no torch, so it must use
    the classic SVM; otherwise default to the most accurate PANNs probe."""
    if FROZEN:
        return "classic"
    if args.classic:
        return "classic"
    # Explicit env override still wins for power users; else best model.
    return os.environ.get("MGC_SERVE_KIND", "embeddings")


def _ensure_cwd() -> None:
    """When frozen by PyInstaller, run from the exe's folder so relative model/
    config paths resolve (we bundle them next to the binary)."""
    if FROZEN:
        os.chdir(Path(sys.executable).parent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Beat Identifier UI.")
    parser.add_argument("--classic", action="store_true",
                        help="serve the light classic SVM instead of the PANNs probe")
    args = parser.parse_args()

    os.environ["MGC_SERVE_KIND"] = _resolve_kind(args)
    _ensure_cwd()

    url = f"http://{HOST}:{PORT}/"
    print(f"\n  Beat Identifier — starting ({os.environ['MGC_SERVE_KIND']} model)…")
    print(f"  Opening {url}\n  (close this window to stop)\n")
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    # Pass the app object directly (frozen apps can't do module-string discovery).
    from src.api.app import app

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
