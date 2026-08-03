"""Transformer turn encoder (Tier 2.1).

Replaces the static-GloVe, flat 5000-step word-level LSTM with a frozen
pretrained transformer that encodes each participant turn independently
(mean-pooled over real tokens), leaving aggregation across turns to a
trainable attention-pooling head (src/models/text_encoder.py) instead of
flattening word-level tokens into one long LSTM sequence.

Model choice: the roadmap's first choice, MentalBERT (mental/mental-bert-
base-uncased), and its AIMH mirror are both gated on Hugging Face without
pre-authorized access. Falls back to distilbert-base-uncased -- ungated,
768-dim, and roughly the RoBERTa-class fallback the roadmap allows.
Swap MODEL_NAME for a mental-health-specific checkpoint if access is
arranged later; nothing else in this module or its callers depends on
which encoder is loaded.
"""
from __future__ import annotations

import numpy as np
import torch

MODEL_NAME = "distilbert-base-uncased"
MAX_TURNS = 250
MAX_TOKEN_LEN = 64

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
        for p in _model.parameters():
            p.requires_grad = False
    return _tokenizer, _model


def hidden_size() -> int:
    _, model = _load()
    return model.config.hidden_size


@torch.no_grad()
def encode_turns(turns: list[str], batch_size: int = 16) -> np.ndarray:
    """(n_turns, hidden_size) mean-pooled embeddings, one per turn."""
    tokenizer, model = _load()
    turns = list(turns)[:MAX_TURNS]
    if not turns:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)

    embeddings = []
    for i in range(0, len(turns), batch_size):
        batch = turns[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_TOKEN_LEN, return_tensors="pt")
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        pooled = (summed / counts).numpy()
        embeddings.append(pooled)
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def pad_turns(embeddings: np.ndarray, max_turns: int = MAX_TURNS) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad/truncate to a fixed number of turns; mask marks real turns."""
    hidden = embeddings.shape[1]
    out = np.zeros((max_turns, hidden), dtype=np.float32)
    mask = np.zeros(max_turns, dtype=np.float32)
    n = min(embeddings.shape[0], max_turns)
    if n > 0:
        out[:n] = embeddings[:n]
        mask[:n] = 1.0
    return out, mask
