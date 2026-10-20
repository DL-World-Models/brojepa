"""Small utilities shared across training scripts."""

import random
import time
import subprocess
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    """Pick the best locally available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for run-to-run reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"

def build_run_dir(root: str, tag: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{ts}-{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
