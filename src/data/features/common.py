"""Shared helpers used by both the text/audio/video feature extractors.

These fix Tier 0 bugs 0.3 (scale_down mutates in place / wrong index),
0.4 (no normalization, train-only statistics), and 0.5 (head/tail
truncation instead of uniform coverage of the interview).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def scale_down(X: np.ndarray, factor: int = 2) -> np.ndarray:
    """Average every `factor` consecutive frames. Pure function, no aliasing."""
    if X.shape[0] < factor:
        return X.copy()
    n = (X.shape[0] // factor) * factor
    return X[:n].reshape(-1, factor, X.shape[1]).mean(axis=1)


def to_fixed_length(X: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample X (T, F) to exactly `target_len` frames, covering the whole
    sequence uniformly instead of slicing from the head or tail.

    - If T >= target_len: bin-average `target_len` equal-width windows spanning
      the full sequence (uniform coverage of the entire interview).
    - If T < target_len: keep all frames and zero-pad at the end.

    Returns (fixed, mask) where mask[t] is 1 for real (non-padded) frames.
    """
    t = X.shape[0]
    if t == 0:
        return np.zeros((target_len, X.shape[1]), dtype=np.float32), np.zeros(target_len, dtype=np.float32)
    if t >= target_len:
        bounds = np.linspace(0, t, target_len + 1).astype(int)
        out = np.empty((target_len, X.shape[1]), dtype=np.float32)
        for i in range(target_len):
            lo, hi = bounds[i], max(bounds[i] + 1, bounds[i + 1])
            out[i] = X[lo:hi].mean(axis=0)
        mask = np.ones(target_len, dtype=np.float32)
        return out, mask
    out = np.zeros((target_len, X.shape[1]), dtype=np.float32)
    out[:t] = X
    mask = np.zeros(target_len, dtype=np.float32)
    mask[:t] = 1.0
    return out, mask


class Scaler:
    """Z-score scaler fit on the training split only, persisted to disk so
    inference uses the exact same statistics as training (Tier 0.4)."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, X: np.ndarray) -> "Scaler":
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        self.std = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fit or loaded before transform()")
        return (X - self.mean) / self.std

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        data = np.load(str(path) if str(path).endswith(".npz") else f"{path}.npz")
        return cls(mean=data["mean"], std=data["std"])


def save_oov_report(path: str | Path, counts: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(counts, indent=2))
