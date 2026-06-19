"""Shared helpers used across every phase: config loading and seeding.

## What this code does
Two small things that the whole project leans on:
1. `load_config` reads a YAML file from `configs/` into a plain dict, so no
   hyperparameter is ever hardcoded in the Python files.
2. `set_seed` pins Python / NumPy / PyTorch random number generators to a fixed
   value so runs are reproducible. (NumPy and PyTorch are imported lazily so
   this module also works in a minimal environment that only has PyYAML.)
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Repo root = two levels up from this file (src/utils.py -> src -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    """Load a YAML config by file name (with or without the .yaml suffix).

    >>> cfg = load_config("data")          # reads configs/data.yaml
    >>> cfg["seed"]
    42
    """
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    path = CONFIGS_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int = 42) -> None:
    """Seed every RNG we use so results are reproducible across runs.

    Imports are lazy: NumPy/PyTorch may not be installed in every environment
    (e.g. when only generating the split manifest), and we don't want to force
    them as a hard dependency of this helper.
    """
    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
