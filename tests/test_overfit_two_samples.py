"""A 2-sample overfit test (Tier 4.2): if the model can't reach ~100%
accuracy on two trivially separable examples, the training loop itself is
broken (wrong loss, wrong concat dim, dead gradients, etc.) independent of
any real-data issues."""
import torch
import torch.nn as nn

from src.models.fusion import MultimodalLSTM, MultimodalTransformerText, TransformerTextOnly
from src.utils import set_seed


def test_multimodal_lstm_overfits_two_examples():
    set_seed(0)
    text_dim, audio_dim, video_dim, hidden = 8, 6, 5, 16
    seq_len = 4

    model = MultimodalLSTM(text_dim, audio_dim, video_dim, hidden, output_size=1, dropout=0.0)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x_text = torch.stack([torch.zeros(seq_len, text_dim), torch.ones(seq_len, text_dim)])
    x_audio = torch.stack([torch.zeros(seq_len, audio_dim), torch.ones(seq_len, audio_dim)])
    x_video = torch.stack([torch.zeros(seq_len, video_dim), torch.ones(seq_len, video_dim)])
    y = torch.tensor([0.0, 1.0])

    for _ in range(200):
        optimizer.zero_grad()
        logits = model(x_text, x_audio, x_video).squeeze(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(model(x_text, x_audio, x_video).squeeze(-1)) > 0.5).float()
    assert torch.equal(preds, y), f"failed to overfit two examples: preds={preds}, loss={loss.item()}"


def test_transformer_text_only_overfits_two_examples():
    """Same sanity check for the Tier 2.1 attention-pooling text head:
    frozen-encoder turn embeddings in, trainable pooling + head only."""
    set_seed(0)
    turns, hidden = 5, 16

    model = TransformerTextOnly(hidden, dropout=0.0)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    text_emb = torch.stack([torch.zeros(turns, hidden), torch.ones(turns, hidden)])
    text_mask = torch.ones(2, turns)
    lexical = torch.stack([torch.zeros(2), torch.ones(2)])
    y = torch.tensor([0.0, 1.0])

    for _ in range(200):
        optimizer.zero_grad()
        logits = model(text_emb, text_mask, lexical).squeeze(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(model(text_emb, text_mask, lexical).squeeze(-1)) > 0.5).float()
    assert torch.equal(preds, y), f"failed to overfit two examples: preds={preds}, loss={loss.item()}"


def test_multimodal_transformer_text_overfits_two_examples():
    set_seed(0)
    turns, hidden, audio_dim, video_dim, lstm_hidden = 5, 16, 6, 5, 16
    seq_len = 4

    model = MultimodalTransformerText(hidden, audio_dim, video_dim, lstm_hidden, dropout=0.0)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    text_emb = torch.stack([torch.zeros(turns, hidden), torch.ones(turns, hidden)])
    text_mask = torch.ones(2, turns)
    lexical = torch.stack([torch.zeros(2), torch.ones(2)])
    x_audio = torch.stack([torch.zeros(seq_len, audio_dim), torch.ones(seq_len, audio_dim)])
    x_video = torch.stack([torch.zeros(seq_len, video_dim), torch.ones(seq_len, video_dim)])
    y = torch.tensor([0.0, 1.0])

    for _ in range(200):
        optimizer.zero_grad()
        logits = model(text_emb, text_mask, lexical, x_audio, x_video).squeeze(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(model(text_emb, text_mask, lexical, x_audio, x_video).squeeze(-1)) > 0.5).float()
    assert torch.equal(preds, y), f"failed to overfit two examples: preds={preds}, loss={loss.item()}"
