"""Shape assertions for feature extraction (Tier 4.2)."""
import numpy as np
import pytest

from src.data.features import text
from src.data.features.common import scale_down, to_fixed_length

DATA_DIR = "daic_woz/data"
PARTICIPANT = "300"
HAVE_DATA = __import__("pathlib").Path(f"{DATA_DIR}/{PARTICIPANT}").exists()


def test_scale_down_shape_and_no_mutation():
    X = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    X_copy = X.copy()
    out = scale_down(X, factor=2)
    assert out.shape == (10, 3)
    np.testing.assert_array_equal(X, X_copy)  # input must not be mutated


def test_scale_down_averages_correct_pairs():
    X = np.array([[0.0], [2.0], [4.0], [8.0]], dtype=np.float32)
    out = scale_down(X, factor=2)
    np.testing.assert_allclose(out, [[1.0], [6.0]])


def test_to_fixed_length_pads_short_sequences():
    X = np.ones((5, 4), dtype=np.float32)
    fixed, mask = to_fixed_length(X, target_len=10)
    assert fixed.shape == (10, 4)
    assert mask.sum() == 5
    assert (fixed[5:] == 0).all()


def test_to_fixed_length_covers_full_sequence_when_downsampling():
    X = np.arange(100, dtype=np.float32).reshape(100, 1)
    fixed, mask = to_fixed_length(X, target_len=10)
    assert fixed.shape == (10, 1)
    assert mask.sum() == 10
    # first bin should average early frames, last bin late frames -- not a head/tail slice
    assert fixed[0, 0] < fixed[-1, 0]
    assert fixed[-1, 0] > 50  # would fail under the old head-only truncation


def test_text_tensor_shape_matches_training_notebook_layout():
    turns = ["hello there", "i am not doing well today"]

    class FakeWV:
        def __getitem__(self, w):
            return np.zeros(text.EMBED_DIM, dtype=np.float32)

    tensor = text.turns_to_tensor(turns, FakeWV())
    assert tensor.shape == (text.MAX_TURNS, text.MAX_WORDS, text.EMBED_DIM)


@pytest.mark.skipif(not HAVE_DATA, reason="daic_woz data not downloaded")
def test_audio_video_column_counts_on_real_participant():
    from src.data.features import audio, video

    araw = audio.extract_raw(f"{DATA_DIR}/{PARTICIPANT}/{PARTICIPANT}_COVAREP.csv")
    assert araw.shape[1] == audio.N_COVAREP

    vraw = video.extract_raw(DATA_DIR, PARTICIPANT)
    assert vraw.ndim == 2
    assert vraw.shape[1] > 0
