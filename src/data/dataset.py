"""Build cached, ID-aligned tensors for every participant in every split.

This is the single place that ties text/audio/video extraction together, so
train.py and prcsfle.py (inference) both go through it -- the root cause of
Tier 0.1 was training and inference having two different implementations of
the same preprocessing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.utils.data import Dataset

from .features import audio, text, text_transformer, video
from .features.common import Scaler
from .loaders import Split, load_splits, participant_data_dir


class ParticipantCache:
    """Caches per-participant raw (pre-normalization) feature matrices to
    disk so repeated runs don't recompute CLNF concatenation / COVAREP
    parsing / word-vector lookups every time."""

    def __init__(self, cache_dir: str, data_dir: str, word_vectors_path: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self._wv_path = word_vectors_path
        self._wv = None
        self.oov_counts: dict[str, int] = {}

    @property
    def word_vectors(self) -> KeyedVectors:
        if self._wv is None:
            self._wv = KeyedVectors.load(self._wv_path)
        return self._wv

    def _raw_path(self, participant_id: str) -> Path:
        return self.cache_dir / f"{participant_id}_raw.npz"

    def raw(self, participant_id: str) -> dict:
        path = self._raw_path(participant_id)
        if path.exists():
            data = np.load(path, allow_pickle=True)
            return {
                "text_tensor": data["text_tensor"],
                "first_person_singular_rate": float(data["first_person_singular_rate"]),
                "negation_rate": float(data["negation_rate"]),
                "turns": list(data["turns"]),
                "audio_raw": data["audio_raw"],
                "video_raw": data["video_raw"],
            }
        text_out = text.extract(
            f"{self.data_dir}/{participant_id}/{participant_id}_TRANSCRIPT.csv",
            self.word_vectors,
            oov_counter=self.oov_counts,
        )
        audio_raw = audio.extract_raw(f"{self.data_dir}/{participant_id}/{participant_id}_COVAREP.csv")
        video_raw = video.extract_raw(self.data_dir, participant_id)
        out = {
            "text_tensor": text_out["tensor"],
            "first_person_singular_rate": text_out["first_person_singular_rate"],
            "negation_rate": text_out["negation_rate"],
            "turns": text_out["turns"],
            "audio_raw": audio_raw,
            "video_raw": video_raw,
        }
        np.savez_compressed(
            path,
            text_tensor=out["text_tensor"],
            first_person_singular_rate=out["first_person_singular_rate"],
            negation_rate=out["negation_rate"],
            turns=np.array(out["turns"], dtype=object),
            audio_raw=out["audio_raw"],
            video_raw=out["video_raw"],
        )
        return out

    def _transformer_path(self, participant_id: str) -> Path:
        return self.cache_dir / f"{participant_id}_text_transformer.npz"

    def transformer_text(self, participant_id: str) -> dict:
        """Frozen transformer turn embeddings (Tier 2.1), cached separately
        from the GloVe word-level tensor since it's a heavier, optional
        dependency most Tier 0/1 runs don't need."""
        path = self._transformer_path(participant_id)
        if path.exists():
            data = np.load(path)
            return {"embeddings": data["embeddings"], "mask": data["mask"]}
        turns = self.raw(participant_id)["turns"]
        emb = text_transformer.encode_turns(list(turns))
        fixed, mask = text_transformer.pad_turns(emb)
        np.savez_compressed(path, embeddings=fixed, mask=mask)
        return {"embeddings": fixed, "mask": mask}


class DDMMLDataset(Dataset):
    """One split (train/dev/test), fully materialized as normalized,
    fixed-length tensors, ID-aligned with labels."""

    def __init__(self, split: Split, cache: ParticipantCache, audio_scaler: Scaler, video_scaler: Scaler,
                 audio_len: int = 1000, video_len: int = 1000, include_transformer_text: bool = False):
        self.split = split
        self.items = []
        for pid in split.participant_ids:
            raw = cache.raw(pid)
            audio_tensor, audio_mask = audio.to_tensor(raw["audio_raw"], audio_scaler, target_len=audio_len)
            video_tensor, video_mask = video.to_tensor(raw["video_raw"], video_scaler, target_len=video_len)
            item = {
                "participant_id": pid,
                "turns": raw["turns"],
                "text": torch.from_numpy(raw["text_tensor"]).view(-1, text.EMBED_DIM).float(),
                "audio": torch.from_numpy(audio_tensor).float(),
                "video": torch.from_numpy(video_tensor).float(),
                "audio_mask": torch.from_numpy(audio_mask).float(),
                "video_mask": torch.from_numpy(video_mask).float(),
                "lexical": torch.tensor([raw["first_person_singular_rate"], raw["negation_rate"]], dtype=torch.float32),
                "label": torch.tensor(float(split.label(pid))),
            }
            if include_transformer_text:
                tt = cache.transformer_text(pid)
                item["text_turns_emb"] = torch.from_numpy(tt["embeddings"]).float()
                item["text_turns_mask"] = torch.from_numpy(tt["mask"]).float()
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def fit_scalers(train_split: Split, cache: ParticipantCache) -> tuple[Scaler, Scaler]:
    audio_raws = [cache.raw(pid)["audio_raw"] for pid in train_split.participant_ids]
    video_raws = [cache.raw(pid)["video_raw"] for pid in train_split.participant_ids]
    return audio.fit_scaler(audio_raws), video.fit_scaler(video_raws)


def build_datasets(config: dict, include_transformer_text: bool = False) -> tuple[dict[str, DDMMLDataset], Scaler, Scaler]:
    splits = load_splits(config)
    data_dir = participant_data_dir(config)
    cache = ParticipantCache(config["data"]["cache_dir"], data_dir, config["data"]["word_vectors"])

    audio_scaler_path = Path(config["checkpoints"]["dir"]) / "audio_scaler"
    video_scaler_path = Path(config["checkpoints"]["dir"]) / "video_scaler"
    if audio_scaler_path.with_suffix(".npz").exists() and video_scaler_path.with_suffix(".npz").exists():
        audio_scaler = Scaler.load(audio_scaler_path)
        video_scaler = Scaler.load(video_scaler_path)
    else:
        audio_scaler, video_scaler = fit_scalers(splits["train"], cache)
        audio_scaler.save(audio_scaler_path)
        video_scaler.save(video_scaler_path)

    audio_len = config["features"]["audio"]["max_frames"]
    video_len = config["features"]["video"]["max_frames"]

    datasets = {
        name: DDMMLDataset(split, cache, audio_scaler, video_scaler, audio_len=audio_len, video_len=video_len,
                            include_transformer_text=include_transformer_text)
        for name, split in splits.items()
    }

    lengths = {name: {"n": len(ds), "audio_dim": ds[0]["audio"].shape[-1], "video_dim": ds[0]["video"].shape[-1]}
               for name, ds in datasets.items()}
    for name, split in splits.items():
        assert len(datasets[name]) == len(split.participant_ids), f"{name}: dataset length != split length"

    if cache.oov_counts:
        oov_path = Path(config["data"]["cache_dir"]) / "oov_report.json"
        oov_path.write_text(json.dumps(dict(sorted(cache.oov_counts.items(), key=lambda kv: -kv[1])[:200]), indent=2))

    print("[dataset] split sizes:", {k: v["n"] for k, v in lengths.items()})
    return datasets, audio_scaler, video_scaler
