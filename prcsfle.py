"""Inference entry point for the Streamlit app.

Fixes Tier 0.1 / 0.7: this used to duplicate its own text/audio/video
preprocessing (text_processing.py, audio_processing.py, video_processing.py)
which had drifted from the training notebook -- most seriously, a swapped
tensor shape that silently truncated every interview to ~20 turns. It now
imports the exact same feature-extraction code the training pipeline uses
(src/data/features/*), so there is only one implementation to keep correct,
and applies the *same* persisted train-split scalers rather than leaving
audio/video features unnormalized.
"""
from __future__ import annotations

import torch
import yaml
from gensim.models import KeyedVectors

from src.data.features import audio, text, video
from src.data.features.common import Scaler
from src.models.fusion import MultimodalLSTM
from src.utils import load_checkpoint

_CONFIG = yaml.safe_load(open("config.yaml"))
_WORD_VECTORS = None
_MODEL = None
_AUDIO_SCALER = None
_VIDEO_SCALER = None


def _word_vectors() -> KeyedVectors:
    global _WORD_VECTORS
    if _WORD_VECTORS is None:
        _WORD_VECTORS = KeyedVectors.load(_CONFIG["data"]["word_vectors"])
    return _WORD_VECTORS


def _scalers() -> tuple[Scaler, Scaler]:
    global _AUDIO_SCALER, _VIDEO_SCALER
    if _AUDIO_SCALER is None:
        ckpt_dir = _CONFIG["checkpoints"]["dir"]
        _AUDIO_SCALER = Scaler.load(f"{ckpt_dir}/audio_scaler")
        _VIDEO_SCALER = Scaler.load(f"{ckpt_dir}/video_scaler")
    return _AUDIO_SCALER, _VIDEO_SCALER


def _model(text_dim: int, audio_dim: int, video_dim: int) -> MultimodalLSTM:
    global _MODEL
    if _MODEL is None:
        model = MultimodalLSTM(
            text_dim, audio_dim, video_dim,
            hidden_size=_CONFIG["model"]["hidden_size"],
            output_size=_CONFIG["model"]["output_size"],
            dropout=_CONFIG["model"]["dropout"],
        )
        load_checkpoint(model, f"{_CONFIG['checkpoints']['dir']}/multimodal_lstm.pth")
        model.eval()
        _MODEL = model
    return _MODEL


def process_pds(transcript, covarep, clnf_au, clnf_feat, clnf_feat3d, clnf_gaze, clnf_pose) -> dict:
    text_out = text.extract(transcript, _word_vectors())
    text_tensor = torch.from_numpy(text_out["tensor"]).view(-1, text.EMBED_DIM).float().unsqueeze(0)

    audio_scaler, video_scaler = _scalers()

    audio_raw = audio.extract_raw(covarep)
    audio_tensor, audio_mask = audio.to_tensor(audio_raw, audio_scaler, target_len=_CONFIG["features"]["audio"]["max_frames"])
    audio_tensor = torch.from_numpy(audio_tensor).float().unsqueeze(0)

    video_raw = video.extract_raw_from_files(clnf_au, clnf_feat, clnf_feat3d, clnf_gaze, clnf_pose)
    video_tensor, video_mask = video.to_tensor(video_raw, video_scaler, target_len=_CONFIG["features"]["video"]["max_frames"])
    video_tensor = torch.from_numpy(video_tensor).float().unsqueeze(0)

    model = _model(text_tensor.shape[-1], audio_tensor.shape[-1], video_tensor.shape[-1])
    with torch.no_grad():
        logits = model(text_tensor, audio_tensor, video_tensor)
        probability = torch.sigmoid(logits).item()

    audio_coverage = float(audio_mask.mean())
    video_coverage = float(video_mask.mean())

    return {
        "probability": probability,
        "first_person_singular_rate": text_out["first_person_singular_rate"],
        "negation_rate": text_out["negation_rate"],
        "n_turns": text_out["n_turns"],
        "audio_coverage": audio_coverage,
        "video_coverage": video_coverage,
    }
