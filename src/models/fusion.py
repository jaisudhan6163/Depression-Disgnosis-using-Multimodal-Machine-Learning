"""Multimodal LSTM fusion model.

Fixes Tier 0.7:
- forward() used to torch.cat(..., dim=0), which only worked for unbatched
  (single-example) input and silently produced garbage for any batch size
  > 1 -- the training notebook used dim=1, so the deployed model and the
  trained weights disagreed about what the concatenated vector meant.
  Unified on dim=1 (batch-first) everywhere.
- forward() used to `return float(output)`, which only works for a single
  scalar and breaks for batched output. Now returns raw logits (a tensor);
  BCEWithLogitsLoss is used at train time instead of sigmoid+BCELoss (also
  numerically safer -- Tier 2.6), and sigmoid is applied explicitly only at
  inference/display time.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .text_encoder import TransformerTextHead


class MultimodalLSTM(nn.Module):
    def __init__(self, text_dim: int, audio_dim: int, video_dim: int, hidden_size: int, output_size: int = 1, dropout: float = 0.3):
        super().__init__()
        self.text_layer = nn.LSTM(input_size=text_dim, hidden_size=hidden_size, batch_first=True)
        self.audio_layer = nn.LSTM(input_size=audio_dim, hidden_size=hidden_size, batch_first=True)
        self.video_layer = nn.LSTM(input_size=video_dim, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 3, output_size)

    def forward(self, x_text: torch.Tensor, x_audio: torch.Tensor, x_video: torch.Tensor) -> torch.Tensor:
        _, (h_text, _) = self.text_layer(x_text)
        _, (h_audio, _) = self.audio_layer(x_audio)
        _, (h_video, _) = self.video_layer(x_video)
        combined = torch.cat((h_text[-1], h_audio[-1], h_video[-1]), dim=1)
        combined = self.dropout(combined)
        logits = self.fc(combined)  # raw logits, no sigmoid here
        return logits


class UnimodalLSTM(nn.Module):
    """Single-modality LSTM used for the audio-only / video-only / text-only
    ablations in Tier 1.3."""

    def __init__(self, input_dim: int, hidden_size: int, output_size: int = 1, dropout: float = 0.3):
        super().__init__()
        self.layer = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.layer(x)
        return self.fc(self.dropout(h[-1]))


class MultimodalTransformerText(nn.Module):
    """Fusion model with the Tier 2.1 transformer + attention-pooling text
    branch (replacing the word-level text LSTM), keeping the audio/video
    LSTM branches and plain-concatenation fusion unchanged -- Tier 2.4
    (attention fusion across modalities) is a separate, not-yet-done
    upgrade."""

    def __init__(self, text_hidden_size: int, audio_dim: int, video_dim: int, hidden_size: int,
                 output_size: int = 1, dropout: float = 0.3, lexical_dim: int = 2):
        super().__init__()
        self.text_head = TransformerTextHead(text_hidden_size, lexical_dim)
        self.text_proj = nn.Linear(self.text_head.output_dim, hidden_size)
        self.audio_layer = nn.LSTM(input_size=audio_dim, hidden_size=hidden_size, batch_first=True)
        self.video_layer = nn.LSTM(input_size=video_dim, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 3, output_size)

    def forward(self, text_turns: torch.Tensor, text_mask: torch.Tensor, lexical: torch.Tensor,
                x_audio: torch.Tensor, x_video: torch.Tensor) -> torch.Tensor:
        text_vec, _ = self.text_head(text_turns, text_mask, lexical)
        text_vec = torch.relu(self.text_proj(text_vec))
        _, (h_audio, _) = self.audio_layer(x_audio)
        _, (h_video, _) = self.video_layer(x_video)
        combined = torch.cat((text_vec, h_audio[-1], h_video[-1]), dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)


class TransformerTextOnly(nn.Module):
    """Text-only ablation using the Tier 2.1 encoder, for direct comparison
    against the Tier 1 word-level-LSTM text-only baseline."""

    def __init__(self, text_hidden_size: int, output_size: int = 1, dropout: float = 0.3, lexical_dim: int = 2):
        super().__init__()
        self.text_head = TransformerTextHead(text_hidden_size, lexical_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.text_head.output_dim, output_size)

    def forward(self, text_turns: torch.Tensor, text_mask: torch.Tensor, lexical: torch.Tensor) -> torch.Tensor:
        vec, _ = self.text_head(text_turns, text_mask, lexical)
        return self.fc(self.dropout(vec))
