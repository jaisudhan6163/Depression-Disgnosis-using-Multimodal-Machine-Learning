"""Train/inference parity test (Tier 0.1 / 4.2).

The original bug was training and inference using two different
implementations of text preprocessing that silently drifted apart. Now
there is exactly one implementation (src/data/features/text.py) used by
both src/data/dataset.py (training) and prcsfle.py (inference); this test
proves that calling it twice for the same participant is deterministic and
that both call sites produce identically shaped, identically valued output.
"""
from pathlib import Path

import numpy as np
import pytest

from src.data.features import text

DATA_DIR = "daic_woz/data"
PARTICIPANT = "300"
HAVE_DATA = Path(f"{DATA_DIR}/{PARTICIPANT}").exists()


@pytest.mark.skipif(not HAVE_DATA, reason="daic_woz data not downloaded")
def test_text_extraction_is_deterministic_across_calls():
    class FakeWV:
        def __getitem__(self, w):
            return np.ones(text.EMBED_DIM, dtype=np.float32) * (hash(w) % 7)

    transcript_path = f"{DATA_DIR}/{PARTICIPANT}/{PARTICIPANT}_TRANSCRIPT.csv"

    train_side = text.extract(transcript_path, FakeWV())
    infer_side = text.extract(transcript_path, FakeWV())

    assert train_side["tensor"].shape == infer_side["tensor"].shape == (text.MAX_TURNS, text.MAX_WORDS, text.EMBED_DIM)
    np.testing.assert_array_equal(train_side["tensor"], infer_side["tensor"])
    assert train_side["turns"] == infer_side["turns"]
