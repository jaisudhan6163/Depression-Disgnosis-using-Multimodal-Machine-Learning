# Depression Screening Support — Multimodal Interview Analysis

A research prototype that estimates depression-screening risk from DAIC-WOZ
clinical interviews, fusing text, audio, and video branches. **This is not a
diagnosis and not a medical device** — PHQ-8 is a self-report screening
instrument, not a clinical diagnosis, and this model's own evaluation
(below) shows it does not generalize to held-out data. See
[Limitations](#limitations).

This repo went through a Tier 0 (correctness) + Tier 1 (honest evaluation) +
Tier 2.1 (transformer text encoder) pass; see [ROADMAP.md](ROADMAP.md) for
the full multi-tier plan this is part of, and
[results/ablations.md](results/ablations.md) for the current numbers and
what they mean.

## Architecture

Two text encoders are available (`config.yaml` / `--transformer-text`),
sharing unchanged audio/video LSTM branches:

- **Text (Tier 1, default)**: participant turns → per-word GloVe-300d
  vectors → fixed (250 turns × 20 words × 300 dims) tensor, flattened to a
  5000-step sequence → LSTM.
- **Text (Tier 2.1, `--transformer-text`)**: participant turns → frozen
  `distilbert-base-uncased`, mean-pooled per turn (768-dim) → trainable
  attention pooling across turns → concatenated with first-person-singular
  rate / negation rate.
- **Audio**: COVAREP (74-dim, 100Hz) → z-scored (train-split stats) →
  resampled to 1000 frames uniformly covering the whole interview → LSTM.
- **Video**: CLNF AU/landmarks/gaze/pose (concatenated, HOG excluded) →
  z-scored → downsampled 2x → resampled to 1000 frames uniformly covering
  the whole interview → LSTM.

→ concatenated final hidden states → `Linear(hidden*3, 1)` → sigmoid.

Baselines: majority class, TF-IDF + logistic regression on transcripts, and
text-only / audio-only / video-only ablations (each text encoder has its
own text-only ablation), all evaluated with the same protocol as fusion.

## Results

| Model | test F1 | test AUROC |
|---|---|---|
| Majority class | 0.000 | 0.500 |
| TF-IDF + LogisticRegression | **0.464** | **0.561** |
| Fusion, GloVe+LSTM text (Tier 1) | 0.372 ± 0.048 | 0.455 |
| Fusion, transformer+attention-pool text (Tier 2.1) | 0.409 ± 0.031 | 0.486 |

**The simple TF-IDF baseline still beats every LSTM/transformer fusion
variant on held-out test AUROC.** The Tier 2.1 text encoder narrows the
dev-to-test overfitting gap (more training data would matter more than a
better text encoder at this point) but doesn't close it — every
LSTM-based model's test AUROC is still at or below 0.5. Full table,
per-seed numbers, and discussion in
[results/ablations.md](results/ablations.md).

## Setup

```bash
pip install -r requirements.txt
```

Requires Python with the packages pinned in `requirements.txt`. Developed
and run against `/opt/anaconda3/bin/python3` (torch 2.11, pandas 2.2,
numpy 2.1, scikit-learn 1.6, gensim 4.4).

### Dataset

DAIC-WOZ is access-restricted; this repo does not include it
(`daic_woz/` is gitignored). This project was built and evaluated against
the Kaggle mirror at
[saifzaman123445/daicwoz](https://www.kaggle.com/datasets/saifzaman123445/daicwoz),
which contains 188 of the 189 official participants (missing 458) and their
TRANSCRIPT/COVAREP/CLNF files (excluding HOG and raw audio, which this
architecture doesn't use and which are too large to store per-participant).
Official split labels (`train_split_Depression_AVEC2017.csv`,
`dev_split_Depression_AVEC2017.csv`, `full_test_split.csv`) came from
separate small label-only Kaggle datasets, since this mirror doesn't ship
them; place them under `daic_woz/labels/` (see `config.yaml`).

Word vectors: GloVe-300d via `gensim.downloader`, saved to
`models/glove.300d.kv` — not the FastText vectors the original version of
this project used (crawl-300d-2M.vec, 4.5GB) or this dataset's own
GoogleNews vectors (3.6GB); neither fit the available disk budget. GloVe-300d
keeps the same 300-dim input the architecture expects. The Tier 2.1
transformer text encoder (`distilbert-base-uncased`, ~270MB) downloads
automatically via `transformers` the first time `--transformer-text` is
used — MentalBERT (the roadmap's first choice) and its AIMH mirror are
both gated on Hugging Face without pre-authorized access; see
`src/data/features/text_transformer.py`.

```bash
python -m src.train                        # Tier 0/1: builds/caches features, trains 5 seeds x 4 model variants
python -m src.train --transformer-text      # also runs the Tier 2.1 transformer-text fusion + text-only variants
pytest tests/                               # shape, train/inference-parity, and overfit sanity tests
streamlit run app.py                        # inference UI (needs a trained checkpoint under models/checkpoints/)
```

## Repo structure

```
src/
  data/
    features/       # text.py, audio.py, video.py, text_transformer.py, common.py -- shared by train.py AND prcsfle.py (inference)
    loaders.py       # official split loading, ID-aligned across modalities
    dataset.py       # caching, Dataset/DataLoader construction
  models/
    fusion.py         # MultimodalLSTM, UnimodalLSTM, MultimodalTransformerText, TransformerTextOnly
    text_encoder.py    # AttentionPool, TransformerTextHead (Tier 2.1)
    baselines.py      # majority class, TF-IDF+LogisticRegression
  train.py            # 5-seed training + evaluation entry point
  evaluate.py          # F1/precision/recall/AUROC/AUPRC, dev-tuned threshold
  utils.py             # seeding, git SHA, safe checkpoint loading
tests/
app.py                 # Streamlit inference UI
prcsfle.py              # inference glue — imports src/data/features (same code path as training)
config.yaml             # all paths/hyperparameters
results/ablations.md    # results table and discussion
```

## What changed from the original version

The original prototype had several correctness bugs that made its reported
numbers meaningless (see [ROADMAP.md](ROADMAP.md) Tier 0 for the full list
with before/after code):

- Training and inference used two different, drifted implementations of
  text preprocessing — inference silently saw ~20 of ~250 interview turns.
  Now there is one shared module (`src/data/features/text.py`) used by both.
- The video pipeline never built dev/test features; only train. Now all
  three splits go through the same loader with an assertion that dataset
  length matches split length.
- `scale_down()` mutated its input in place through a numpy view and
  averaged the wrong frame pairs. Replaced with a pure, vectorized version.
- No normalization anywhere, and a dropped-index bug left raw ascending
  **timestamp** as a model feature. Now every continuous feature is
  z-scored with train-split-only statistics (persisted and reused at
  inference), and confidence is kept as an explicit unnormalized mask
  rather than folded into the feature vector.
- Audio kept the **last** 1000 frames (final ~10s); video kept the
  **first** 1000 raw frames (first ~67s) — non-overlapping slices of a
  ~15-minute interview. Both now uniformly resample the entire interview.
- NLTK stopword removal deleted first-person pronouns and negations —
  among the most replicated linguistic markers of depression. Removed;
  first-person-singular rate and negation rate are now computed as
  explicit engineered features.
- Missing-label participants were silently scored "not depressed" instead
  of raising. `Split.label()` now raises.
- `torch.load()` without `weights_only=True` (arbitrary code execution via
  pickle); inference used `torch.cat(dim=0)` while training used `dim=1`
  (only worked unbatched). Both fixed; seeds and git SHA are recorded per run.
- No train/dev/test discipline: dev was folded into training and test
  accuracy was printed every epoch (test-set peeking). Now the model is
  selected on dev only, and test is touched once per seed after selection.
- Accuracy was the only reported metric on a ~70/30-imbalanced dataset,
  where a constant predictor scores ~70%. Now F1/precision/recall/AUROC/
  AUPRC/confusion matrix, with a dev-tuned decision threshold.
- No baselines existed. Majority-class and TF-IDF+LogisticRegression are
  now run with the same protocol — and beat the LSTM (see Results above).

**Tier 2.1** replaced the static-GloVe, flat 5000-step word-level text LSTM
with a frozen pretrained transformer encoding each turn independently,
aggregated by a trainable attention-pooling head — narrows but does not
close the dev/test overfitting gap (see Results above and
[results/ablations.md](results/ablations.md)).

## Limitations

- **n=188** (107/34/47 train/dev/test), a single US-based sample from a
  Wizard-of-Oz interview setting that does not resemble clinical intake.
  Gender-imbalanced (see `daic_woz/labels/*.csv`); performance may not be
  uniform across gender, dialect, culture, or age.
- The dataset is a third-party Kaggle mirror, not the official USC ICT
  distribution — verify licensing before any use beyond research/education.
- **This model does not currently generalize**: test AUROC for every
  LSTM/transformer fusion or unimodal variant is at or below chance,
  despite reasonable dev performance. Only the TF-IDF baseline clears
  chance on test. Do not treat any of these numbers as usable for
  screening decisions.
- The displayed "risk score" is a raw sigmoid output, not a calibrated
  probability (no Platt scaling / isotonic regression has been applied).
- This pass (Tier 0 + Tier 1 + Tier 2.1) fixed correctness, measurement,
  and the text encoder; it did not address the remaining Tier 2 modeling
  upgrades (eGeMAPS/wav2vec audio, CLNF-derived behavioral-stats or
  OpenFace video, attention fusion across modalities, PHQ-8 regression) or
  the cross-corpus generalization work in Tier 3 — the results above
  suggest more training data matters more than further architecture
  changes at this sample size, which no Tier 2/3 item on its own fixes.
- The Tier 2.1 text encoder is `distilbert-base-uncased`, not the
  roadmap's preferred mental-health-specific MentalBERT (gated, no
  pre-authorized access) — see `src/data/features/text_transformer.py`.
