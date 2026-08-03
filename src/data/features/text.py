"""Text feature extraction — shared by training and inference (fixes Tier 0.1).

Bug 0.1: the old text_processing.py built (20, 250, 300) while the training
notebook built (N, 250, 20, 300); the deployed module silently discarded
every turn past index 19 because IndexError was swallowed by a bare except.
Here there is exactly one function that both train.py and prcsfle.py import,
so drift between them is no longer possible.

Bug 0.6: NLTK's English stoplist removes first-person singular pronouns
(I, me, my) and negations (not, no, never) -- among the most replicated
linguistic markers of depression. Stopword filtering is removed entirely;
lexical_features() exposes first-person-singular and negation rate as
explicit engineered features for the baselines.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

MAX_TURNS = 250
MAX_WORDS = 20
EMBED_DIM = 300

FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"}
NEGATIONS = {"not", "no", "nothing", "never", "none", "nobody", "nowhere", "neither", "nor", "n't", "cannot", "can't", "won't", "don't", "didn't", "isn't", "wasn't"}

_TAG_RE = re.compile(r"^<(.*?)>?$")


def clean_token(word: str) -> str:
    """Strip DAIC-WOZ transcript markup like <sigh> or <sigh (no closing bracket)."""
    if not word:
        return word
    if word[0] == "<":
        m = _TAG_RE.match(word)
        return m.group(1) if m else word[1:]
    if word.endswith(">"):
        return word[:-1]
    return word


def load_participant_turns(transcript_path) -> list[str]:
    """Return the list of Participant utterances, in order, from a
    DAIC-WOZ *_TRANSCRIPT.csv file."""
    df = pd.read_csv(transcript_path, delimiter="\t", encoding="utf-8", engine="python")
    turns = []
    for _, row in df.iterrows():
        if row.get("speaker") == "Participant":
            value = row.get("value")
            if isinstance(value, str):
                turns.append(value)
    return turns


def turns_to_tensor(turns: list[str], word_vectors, oov_counter: dict | None = None) -> np.ndarray:
    """(MAX_TURNS, MAX_WORDS, EMBED_DIM) float32 tensor. No stopword removal."""
    matrix = np.zeros((MAX_TURNS, MAX_WORDS, EMBED_DIM), dtype=np.float32)
    for i in range(min(MAX_TURNS, len(turns))):
        words = turns[i].split(" ")
        for j in range(min(MAX_WORDS, len(words))):
            word = clean_token(words[j])
            if not word:
                continue
            try:
                matrix[i, j] = word_vectors[word]
            except KeyError:
                if oov_counter is not None:
                    oov_counter[word] = oov_counter.get(word, 0) + 1
    return matrix


def lexical_features(turns: list[str]) -> dict:
    """First-person-singular rate and negation rate over all words spoken
    by the participant (Tier 0.6). Computed on raw words, unfiltered."""
    total = 0
    fps = 0
    neg = 0
    for turn in turns:
        for raw in turn.split(" "):
            word = clean_token(raw).lower().strip(".,!?;:")
            if not word:
                continue
            total += 1
            if word in FIRST_PERSON_SINGULAR:
                fps += 1
            if word in NEGATIONS:
                neg += 1
    if total == 0:
        return {"first_person_singular_rate": 0.0, "negation_rate": 0.0, "n_words": 0, "n_turns": len(turns)}
    return {
        "first_person_singular_rate": fps / total,
        "negation_rate": neg / total,
        "n_words": total,
        "n_turns": len(turns),
    }


def extract(transcript_path, word_vectors, oov_counter: dict | None = None) -> dict:
    """Full extraction for one participant: word tensor + engineered features."""
    turns = load_participant_turns(transcript_path)
    tensor = turns_to_tensor(turns, word_vectors, oov_counter=oov_counter)
    lex = lexical_features(turns)
    return {"tensor": tensor, "turns": turns, **lex}
