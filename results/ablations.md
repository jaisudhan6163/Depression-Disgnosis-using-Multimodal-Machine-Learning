# Results — Tier 0 + Tier 1 (correctness fixes + honest evaluation)

Run: `python -m src.train` · git SHA `cae76610b778cb20c24d14bd991590969bb9313b` · 5 seeds (0-4) ·
threshold tuned on dev per seed, test touched once per seed after selection.

## Dataset

The public Kaggle mirror ([saifzaman123445/daicwoz](https://www.kaggle.com/datasets/saifzaman123445/daicwoz))
contains 188 of the 189 official DAIC-WOZ participants (missing 458). After
intersecting with the official AVEC2017 split labels:

| Split | n | positive | rate |
|---|---|---|---|
| train | 107 | 33 | 31% |
| dev | 34 | 12 | 35% |
| test | 47 | 14 | 30% |

This is effectively the full official split (train and test are exactly
official size; dev is missing one participant). HOG features and raw audio
were not downloaded (unused by this architecture, and disk-prohibitive:
~800MB/participant). Word vectors are GloVe-300d (gensim), not the FastText
vectors named in the original README — the original crawl-300d-2M.vec
(4.5GB) and this dataset's own GoogleNews vectors (3.6GB) didn't fit the
available disk budget; GloVe-300d keeps the embedding dimension unchanged.

## Results (F1 is the depressed/positive class; mean ± std over 5 seeds where applicable)

| Model | dev F1 | dev AUROC | test F1 | test P | test R | test AUROC | test AUPRC |
|---|---|---|---|---|---|---|---|
| Majority class | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 0.298 |
| TF-IDF + LogisticRegression | 0.533 | 0.587 | **0.464** | 0.310 | 0.929 | **0.561** | 0.353 |
| Text-only LSTM | 0.524 ± 0.005 | 0.487 | 0.459 ± 0.000 | 0.298 | 1.000 | 0.555 | 0.349 |
| Audio-only LSTM | 0.650 ± 0.015 | 0.706 | 0.375 ± 0.024 | 0.302 | 0.514 | 0.489 | 0.354 |
| Video-only LSTM | 0.626 ± 0.024 | 0.649 | 0.359 ± 0.089 | 0.287 | 0.500 | 0.471 | 0.324 |
| **Fusion LSTM (fixed architecture)** | 0.647 ± 0.026 | 0.639 | 0.360 ± 0.048 | 0.288 | 0.514 | 0.474 | 0.358 |

Per-seed detail and confusion matrices: `results/metrics.json`.

## The finding

**TF-IDF + logistic regression beats every LSTM variant on held-out test
AUROC**, including the multimodal fusion model this repo is built around.
All three LSTM models look competitive or better on dev (F1 up to 0.65,
AUROC up to 0.71) but their test AUROC clusters at or below 0.5 —
no better than chance on data they didn't get to influence hyperparameter
selection on. That gap between dev and test performance is overfitting:
~250k LSTM parameters (fusion) trained on 107 examples, exactly the failure
mode the roadmap warned about in Tier 0's guiding principle.

This is the "honest floor" the roadmap's Tier 0/1 phase is meant to
produce, not a finished result. The text-only model's R=1.000 / F1=0.459 on
test is a degenerate always-mostly-positive predictor (consistent with its
dev AUROC of 0.487, worse than chance) — it is not "the best text model,"
it's a symptom of the same overfitting on a small, class-imbalanced sample.

Per Tier 2, the fix is architectural (turn-level attention pooling instead
of a flat 5000-step word-level LSTM, eGeMAPS/wav2vec audio, OpenFace video,
stronger regularization/fewer parameters) — out of scope for this Tier 0+1
pass, which targeted correctness and measurement, not modeling.
