# Results

Run: `python -m src.train --transformer-text` · git SHA `0517ce2f56c75bbed19e88a58a40a1ba3a5adbe0`
(plus the Tier 2.1 transformer-text commit on top) · 5 seeds (0-4) · threshold
tuned on dev per seed, test touched once per seed after selection.

## Dataset

The public Kaggle mirror ([saifzaman123445/daicwoz](https://www.kaggle.com/datasets/saifzaman123445/daicwoz))
contains 188 of the 189 official DAIC-WOZ participants (missing 458). After
intersecting with the official AVEC2017 split labels:

| Split | n | positive | rate |
|---|---|---|---|
| train | 107 | 33 | 31% |
| dev | 34 | 12 | 35% |
| test | 47 | 14 | 30% |

HOG features and raw audio were not downloaded (unused by this
architecture, and disk-prohibitive: ~800MB/participant). Word vectors for
the Tier 0/1 GloVe-LSTM text branch are GloVe-300d (gensim); the Tier 2.1
transformer text branch uses `distilbert-base-uncased` (see below).

## Results (F1 is the depressed/positive class; mean ± std over 5 seeds where applicable)

| Model | dev F1 | dev AUROC | test F1 | test P | test R | test AUROC | test AUPRC |
|---|---|---|---|---|---|---|---|
| Majority class | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 0.298 |
| TF-IDF + LogisticRegression | 0.533 | 0.587 | **0.464** | 0.310 | 0.929 | **0.561** | 0.353 |
| Text-only, GloVe+LSTM (Tier 1) | 0.526 ± 0.006 | 0.414 | 0.461 ± 0.003 | 0.299 | 1.000 | 0.484 | 0.299 |
| **Text-only, transformer+attention-pool (Tier 2.1)** | 0.580 ± 0.027 | 0.539 | 0.470 ± 0.024 | 0.314 | 0.943 | 0.461 | 0.291 |
| Audio-only LSTM | 0.646 ± 0.042 | 0.628 | 0.336 ± 0.079 | 0.266 | 0.557 | 0.459 | 0.320 |
| Video-only LSTM | 0.599 ± 0.020 | 0.538 | 0.409 ± 0.094 | 0.306 | 0.629 | 0.525 | 0.336 |
| Fusion, GloVe+LSTM text (Tier 1) | 0.652 ± 0.024 | 0.689 | 0.372 ± 0.048 | 0.326 | 0.514 | 0.455 | 0.330 |
| **Fusion, transformer+attention-pool text (Tier 2.1)** | 0.655 ± 0.041 | 0.642 | 0.409 ± 0.031 | 0.298 | 0.671 | 0.486 | 0.325 |

Per-seed detail and confusion matrices: `results/metrics.json`.

## Tier 2.1 — transformer text encoder

Per `ROADMAP.md` 2.1: replaced the static-GloVe, flat 5000-step word-level
LSTM (250 turns x 20 words flattened into one sequence) with a **frozen
pretrained transformer encoding each turn independently**, aggregated by a
**trainable attention-pooling head** — the encoder does no learning at
all; only the attention query and a small linear head are trained.

**Model substitution.** The roadmap's first choice, MentalBERT
(`mental/mental-bert-base-uncased`), and its `AIMH` mirror are both gated
on Hugging Face without pre-authorized access (403 on download). Used
`distilbert-base-uncased` instead — ungated, 768-dim, comparable in spirit
to the roadmap's explicitly-sanctioned RoBERTa fallback. `MODEL_NAME` in
`src/data/features/text_transformer.py` is the only place this would need
to change if mental-health-specific weights become available.

**What improved:**
- Text-only test F1 0.461 → 0.470, dev F1 0.526 → 0.580 (higher and more
  seed-to-seed variation — the GloVe-LSTM had collapsed to a near-identical
  always-predict-positive classifier every seed, recall exactly 1.000; the
  transformer version is less degenerate, recall 0.943).
- Fusion test F1 0.372 → 0.409, test AUROC 0.455 → 0.486 — closer to (but
  still under) chance.

**What did not improve:** every LSTM-based model, with either text
encoder, still has **test AUROC at or below 0.5** — no better than a coin
flip on data that didn't influence model selection. TF-IDF + logistic
regression (test AUROC 0.561) remains the single best-performing model,
by a clear margin, over both the Tier 1 and Tier 2.1 architectures. A
better text encoder narrows the overfitting gap; it does not close it.
With ~107 training examples, the audio and video LSTM branches (and the
fusion model's fc layer) are still the dominant source of the
train/dev-to-test generalization failure documented in Tier 0/1 — Tier
2.2 (eGeMAPS/wav2vec audio) and 2.3 (behavioral-stats video) are the next
candidates for the same treatment, followed by 2.4 (attention fusion +
modality dropout), none of which have been done yet.
