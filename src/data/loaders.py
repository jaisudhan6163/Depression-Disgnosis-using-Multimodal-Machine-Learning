"""Official train/dev/test split loading, ID-aligned across modalities.

Fixes:
- 0.2: video features are now built for all three splits (the notebook only
  ever built train_split).
- 0.7: checkPosNe used to return 0 (== "not depressed") for any participant
  ID it couldn't find, silently mislabeling missing participants. Split.label()
  raises KeyError instead.
- Every split is built from one ordered participant-ID list, so text/audio/
  video/label are guaranteed to correspond to the same participant at the
  same index (the mismatched-length bug in 0.2 came from *not* doing this).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Split:
    name: str
    participant_ids: list[str]
    labels: dict[str, int]

    def label(self, participant_id: str) -> int:
        if participant_id not in self.labels:
            raise KeyError(f"No PHQ8_Binary label for participant {participant_id} in split '{self.name}'")
        return self.labels[participant_id]


def _read_split_csv(path) -> dict[str, int]:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Participant_ID"])
    labels = {}
    for _, row in df.iterrows():
        pid = str(int(row["Participant_ID"]))
        labels[pid] = int(row["PHQ8_Binary"])
    return labels


def _available_participants(data_root: str | Path) -> set[str]:
    data_root = Path(data_root)
    return {p.name for p in data_root.iterdir() if p.is_dir()}


def load_splits(config: dict) -> dict[str, Split]:
    """Build train/dev/test Split objects, intersected with participants that
    actually have downloaded sensor data. Raises if a split ends up empty."""
    data_dir = Path(config["data"]["root"]) / "data"
    available = _available_participants(data_dir)

    csvs = {
        "train": config["data"]["train_split_csv"],
        "dev": config["data"]["dev_split_csv"],
        "test": config["data"]["test_split_csv"],
    }

    splits = {}
    for name, csv_path in csvs.items():
        labels = _read_split_csv(csv_path)
        missing = [pid for pid in labels if pid not in available]
        ids = sorted((pid for pid in labels if pid in available), key=int)
        if missing:
            print(f"[loaders] split '{name}': {len(missing)} labeled participant(s) have no local data, skipping: {missing}")
        if not ids:
            raise RuntimeError(f"Split '{name}' has zero participants with both a label and local data.")
        splits[name] = Split(name=name, participant_ids=ids, labels=labels)

    train_ids = set(splits["train"].participant_ids)
    dev_ids = set(splits["dev"].participant_ids)
    test_ids = set(splits["test"].participant_ids)
    overlap = (train_ids & dev_ids) | (train_ids & test_ids) | (dev_ids & test_ids)
    if overlap:
        raise RuntimeError(f"Participant ID(s) appear in more than one split -- leakage: {overlap}")

    return splits


def participant_data_dir(config: dict) -> str:
    return str(Path(config["data"]["root"]) / "data")
