"""Desktop launcher: boot the FastAPI server and open the UI in the browser.

This is the entry point bundled into the Windows executable (see build_exe.py).
Run it directly with `python run_ui.py`, or double-click the built .exe.

By default it serves the light **classic SVM** model so the app starts instantly
and packages into a small executable (no torch / PANNs download). To serve the
more accurate PANNs probe instead, set MGC_SERVE_KIND=embeddings before running.
"""

from __future__ import annotations

import os
import sys
import threading
import webbrowser
from pathlib import Path

HOST = os.environ.get("MGC_HOST", "127.0.0.1")
PORT = int(os.environ.get("MGC_PORT", "8000"))
# A bundled exe ships only the light model; default to it unless overridden.
os.environ.setdefault("MGC_SERVE_KIND", "classic")


def _ensure_cwd() -> None:
    """When frozen by PyInstaller, run from the exe's folder so relative model/
    config paths resolve (we bundle them next to the binary)."""
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).parent)


def main() -> None:
    _ensure_cwd()
    url = f"http://{HOST}:{PORT}/"
    print(f"\n  Music Genre Classifier — starting…\n  Opening {url}\n  (close this window to stop)\n")
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    # Import string would require module discovery from the frozen app; pass the
    # app object directly instead (no --reload in a packaged build).
    from src.api.app import app

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
