"""Attention pooling over frozen transformer turn embeddings (Tier 2.1).

Replaces flattening 250x20 word-level steps into one LSTM: each turn is
already a single vector (from the frozen transformer encoder), and this
module learns which turns matter via a trainable attention query -- the
only trainable text-side parameters, since the encoder itself is frozen.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, turns, hidden), mask: (batch, turns) with 1 for real turns
        scores = self.query(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class TransformerTextHead(nn.Module):
    """Frozen-encoder turn embeddings -> attention pooling -> pooled vector
    concatenated with the Tier 0.6 lexical features (first-person-singular
    rate, negation rate)."""

    def __init__(self, hidden_size: int, lexical_dim: int = 2):
        super().__init__()
        self.pool = AttentionPool(hidden_size)
        self.output_dim = hidden_size + lexical_dim

    def forward(self, turn_embeddings: torch.Tensor, turn_mask: torch.Tensor, lexical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, weights = self.pool(turn_embeddings, turn_mask)
        return torch.cat([pooled, lexical], dim=1), weights
