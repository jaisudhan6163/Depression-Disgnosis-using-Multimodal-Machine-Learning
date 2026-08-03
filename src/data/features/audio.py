"""Audio (COVAREP) feature extraction — shared by training and inference.

Fixes:
- 0.4: z-score every continuous feature using train-split statistics only
  (the VUV flag itself is left unnormalized/untouched as a mask-like signal).
- 0.5: the old pipeline kept the *last* 1000 frames (final ~10s). That is a
  disjoint time window from the video pipeline's *first* 1000 frames. Both
  now use `to_fixed_length`, which uniformly covers the full interview.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Scaler, to_fixed_length

N_COVAREP = 74
VUV_COL = 1  # voiced/unvoiced flag: 1 if voiced, 0 if unvoiced


def zero_unvoiced_pitch_features(X: np.ndarray) -> np.ndarray:
    """When a frame is unvoiced (VUV==0), pitch/polarity features 0-7 are
    undefined in COVAREP and are zeroed rather than left as extrapolated
    junk. Pure function -- returns a new array, no aliasing."""
    X = X.copy()
    unvoiced = X[:, VUV_COL] == 0
    X[unvoiced, 0:8] = 0
    return X


def load_covarep(path) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :].values.astype(np.float32)
    return zero_unvoiced_pitch_features(X)


def extract_raw(path) -> np.ndarray:
    """Raw (T, 74) COVAREP matrix for one participant, cleaned but not yet
    resampled or normalized (normalization needs train-split statistics)."""
    return load_covarep(path)


def fit_scaler(raw_matrices: list[np.ndarray]) -> Scaler:
    """Fit z-score stats on concatenated train-split frames, excluding the
    VUV flag column which stays a raw 0/1 mask."""
    concat = np.concatenate(raw_matrices, axis=0)
    scaler = Scaler().fit(concat)
    scaler.mean[VUV_COL] = 0.0
    scaler.std[VUV_COL] = 1.0
    return scaler


def to_tensor(raw: np.ndarray, scaler: Scaler, target_len: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    normalized = scaler.transform(raw)
    fixed, mask = to_fixed_length(normalized, target_len)
    return fixed.astype(np.float32), mask
