"""Video (CLNF) feature extraction — shared by training and inference.

Fixes:
- 0.3: scale_down() used to index a *view* (`X[i*size]`) and mutate it via
  `+=`, corrupting subsequent reads, while also indexing `X[i+j]` instead of
  `X[i*size+j]` (averaging the wrong frames). Replaced with a pure,
  vectorized implementation (common.scale_down).
- 0.4: processData() dropped frame index and then confidence (column 1 of
  the *result*), leaving raw ascending timestamp as a feature. Now frame
  index, timestamp, and success are all dropped; confidence is kept
  separately as an explicit mask channel, never z-scored; every remaining
  continuous feature is z-scored with train-split statistics.
- 0.5: first-1000-raw-frames truncation (~67s) replaced with
  `to_fixed_length`, uniformly covering the whole interview.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Scaler, scale_down, to_fixed_length

FILE_SUFFIXES = ["CLNF_AUs.txt", "CLNF_features.txt", "CLNF_features3D.txt", "CLNF_gaze.txt", "CLNF_pose.txt"]
# Feature count is derived from the data at runtime (see dataset.py), not hardcoded here --
# it depends on the exact CLNF file variant and was wrong (388) in the original codebase.


def load_one(path) -> tuple[np.ndarray, np.ndarray]:
    """Load one CLNF_*.txt file. Returns (features, confidence) with
    frame index, timestamp, and success dropped. Rows with failed tracking
    (confidence == 0 or any non-numeric value) are zeroed."""
    df = pd.read_csv(path, delimiter=",", engine="python")
    df.columns = [c.strip() for c in df.columns]
    values = df.apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    confidence = values[:, 2]
    features = values[:, 4:]  # drop frame(0), timestamp(1), confidence(2), success(3)
    bad_tracking = np.isnan(features).any(axis=1) | (confidence == 0) | np.isnan(confidence)
    features = np.nan_to_num(features, nan=0.0)
    features[bad_tracking] = 0.0
    confidence = np.nan_to_num(confidence, nan=0.0)
    confidence[bad_tracking] = 0.0
    return features, confidence


def concat_sources(au, feat, feat3d, gaze, pose) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate AU/features/features3D/gaze/pose from five already-opened
    sources (file paths or file-like objects, e.g. Streamlit uploads).
    HOG is intentionally excluded (unused by this architecture, and far too
    large to store: 350-450MB per participant)."""
    feats, confidences = [], []
    for src in (au, feat, feat3d, gaze, pose):
        f, c = load_one(src)
        feats.append(f)
        confidences.append(c)
    n = min(x.shape[0] for x in feats)
    feats = [x[:n] for x in feats]
    confidence = np.mean([c[:n] for c in confidences], axis=0)
    return np.concatenate(feats, axis=1), confidence


def load_participant(data_dir, participant_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate AU/features/features3D/gaze/pose for one participant on disk."""
    paths = [f"{data_dir}/{participant_id}/{participant_id}_{suf}" for suf in FILE_SUFFIXES]
    return concat_sources(*paths)


def extract_raw(data_dir, participant_id: str) -> np.ndarray:
    """Raw (T, N_FEATURES) matrix after scale_down, before normalization."""
    features, _confidence = load_participant(data_dir, participant_id)
    return scale_down(features, factor=2)


def extract_raw_from_files(au, feat, feat3d, gaze, pose) -> np.ndarray:
    """Same as extract_raw but for five already-opened file-like sources
    (used by the inference path, which receives Streamlit uploads)."""
    features, _confidence = concat_sources(au, feat, feat3d, gaze, pose)
    return scale_down(features, factor=2)


def fit_scaler(raw_matrices: list[np.ndarray]) -> Scaler:
    concat = np.concatenate(raw_matrices, axis=0)
    return Scaler().fit(concat)


def to_tensor(raw: np.ndarray, scaler: Scaler, target_len: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    normalized = scaler.transform(raw)
    fixed, mask = to_fixed_length(normalized, target_len)
    return fixed.astype(np.float32), mask
