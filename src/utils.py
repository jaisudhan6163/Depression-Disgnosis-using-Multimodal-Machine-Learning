"""Reproducibility helpers (Tier 0.7): seed everything, record the git SHA
that produced a given run, and load checkpoints safely."""
from __future__ import annotations

import random
import subprocess

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_checkpoint(model: torch.nn.Module, path: str) -> None:
    # weights_only=True (Tier 0.7): torch.load without it is an arbitrary
    # code execution path via pickle.
    state = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(state)
