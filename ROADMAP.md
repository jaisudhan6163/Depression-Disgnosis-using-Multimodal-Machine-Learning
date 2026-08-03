# Depression Screening from Multimodal Interviews — Project Roadmap

A prioritized plan for taking `depression-diagnosis-multimodal-machine-learning` from a working prototype to a defensible research project.

**Current architecture:** DAIC-WOZ → three parallel single-layer LSTMs (text 300-d fastText, audio 74-d COVAREP, video 388-d CLNF) → concatenated final hidden states → `Linear(192, 1)` → sigmoid → Streamlit UI.

**Guiding principle:** fix correctness before adding capability. Several bugs below invalidate the current numbers, so any improvement measured against them is measuring noise.

---

## Table of contents

- [Tier 0 — Correctness bugs](#tier-0--correctness-bugs)
- [Tier 1 — Evaluation protocol](#tier-1--evaluation-protocol)
- [Tier 2 — Modeling upgrades](#tier-2--modeling-upgrades)
- [Tier 3 — Generalization](#tier-3--generalization)
- [Tier 4 — Engineering and product](#tier-4--engineering-and-product)
- [Tier 5 — Framing and ethics](#tier-5--framing-and-ethics)
- [Suggested sequencing](#suggested-sequencing)
- [Target repo structure](#target-repo-structure)
- [Reference numbers](#reference-numbers)

---

## Tier 0 — Correctness bugs

> Do these before anything else. Until they're fixed, model comparisons are meaningless.

### 0.1 Training and inference use different text layouts — **critical**

**File:** `text_processing.py` vs `ddmml_training.ipynb` (cell 5)

The notebook builds `finalMatrix` as `(N, 250, 20, 300)` — 250 sentences × 20 words. The deployed module builds `(20, 250, 300)` but the loops still iterate sentences to 250 and words to 20. Every write past sentence index 19 raises `IndexError`, gets swallowed by the bare `except: continue`, and is silently discarded.

**Effect:** the deployed model sees ~20 turns of a ~250-turn interview, laid out differently from training.

```python
# BEFORE — text_processing.py
finalMatrix = np.zeros((20, 250, 300))          # dims swapped
...
except Exception as e:
    continue                                     # hides IndexError

# AFTER
MAX_SENTENCES, MAX_WORDS, EMB_DIM = 250, 20, 300
finalMatrix = np.zeros((MAX_SENTENCES, MAX_WORDS, EMB_DIM), dtype=np.float32)
...
except KeyError:            # only catch out-of-vocabulary
    oov_count += 1
    continue
```

- [x] Extract preprocessing into a single shared module imported by **both** training and inference. Duplicated logic is how this drifted apart.
- [x] Replace every bare `except:` with a specific exception; log and count what you swallow.
- [x] Add a smoke test asserting `train_tensor.shape == inference_tensor.shape` for one fixed participant.

### 0.2 Video pipeline never processed dev or test — **critical**

**File:** `ddmml_training.ipynb` cells 15–16

Only `train_split` is built; the dev and test calls are commented out. `X_video_train` has 107 rows while text/audio have 142, and `X_video_test` is never defined. The training loop iterates `range(0, 142, batch_size)` across all three tensors, producing mismatched batch dimensions in `torch.cat(..., dim=1)`.

- [x] Build video features for all three splits.
- [x] Assert equal lengths across modalities before training: `assert len(X_text) == len(X_audio) == len(X_video) == len(Y)`.
- [x] Assert participant-ID alignment, not just length — build all modalities from one ordered ID list.

### 0.3 `scale_down()` corrupts data in place

**Files:** `video_processing.py`, notebook cell 14

`cur_row = X[i*size]` returns a **view**, so `cur_row += X[i+j]` writes back into `X`, corrupting subsequent reads. It also indexes `X[i+j]` instead of `X[i*size+j]`, averaging frame `2i` with frame `i+1`.

```python
# BEFORE
def scale_down(X):
    X_new = []
    size = 2
    for i in range(int(X.shape[0] / size)):
        cur_row = X[i * size]          # view, not copy
        for j in range(1, size):
            if i + j < X.shape[0]:
                cur_row += X[i + j]    # wrong index AND mutates X
        cur_row = cur_row / size
        X_new.append(cur_row)
    return np.array(X_new)

# AFTER
def scale_down(X, factor=2):
    n = (X.shape[0] // factor) * factor
    return X[:n].reshape(-1, factor, X.shape[1]).mean(axis=1)
```

### 0.4 No normalization anywhere

COVAREP mixes F0 in Hz (~100–300), VUV as 0/1, and NAQ (~0.1). CLNF mixes pixel coordinates (hundreds), gaze in radians, and AU intensities (0–5). Unnormalized, the largest-magnitude features dominate the LSTM gates.

Worse — `processData` deletes column 0 (frame index), then column 1 *of the result* (confidence), leaving **timestamp** as a feature. Timestamp is monotonically increasing raw seconds and will swamp everything.

- [x] Drop both frame index and timestamp from the feature matrix.
- [x] Keep `confidence` / `success` as an explicit **mask channel**, not as a feature to be normalized.
- [x] Z-score every continuous feature using **training-split statistics only**. Persist the scaler (`joblib`) and load it at inference.
- [ ] For failed-tracking frames, prefer masking + packed sequences over zero-filling — the model currently can't distinguish "zero" from "missing".

### 0.5 Severe and inconsistent truncation

| Modality | Rate | Kept | Actual coverage |
|---|---|---|---|
| Audio (COVAREP) | 100 Hz | **last** 1000 frames | final ~10 seconds |
| Video (CLNF) | 30 fps, ÷2 | **first** 1000 frames | first ~67 seconds |
| Text | per turn | first 20 turns (deployed) | opening few minutes |

The three modalities are looking at three largely non-overlapping slices of a ~15-minute interview. Nothing can fuse meaningfully across that.

- [ ] Align all modalities to the **same time window** using transcript turn timestamps.
- [ ] Prefer segment-level aggregation (per participant turn) over fixed-length raw-frame truncation.
- [x] If you keep truncation, sample uniformly across the full interview rather than head/tail slicing.

### 0.6 Stopword removal deletes the strongest signal

First-person singular pronouns (`I`, `me`, `my`, `myself`) and negations (`not`, `no`, `nothing`, `never`) are among the most robustly replicated linguistic markers of depression — and NLTK's English stoplist removes all of them.

- [x] Remove stopword filtering entirely for this task.
- [x] Add first-person-singular rate and negation rate as explicit engineered features.

### 0.7 Miscellaneous correctness

- [x] `checkPosNeg()` returns `0` for unmatched IDs — silently mislabels missing participants as non-depressed. Raise instead.
- [x] `prcsfle.py` uses `torch.cat(..., dim=0)` and `float(output)` while the notebook uses `dim=1`. The inference path only works unbatched. Unify on `dim=1` and keep tensors.
- [x] `torch.load(...)` without `weights_only=True` is an arbitrary-code-execution path. Set it.
- [x] Set and record seeds for `random`, `numpy`, and `torch`; log the git SHA with every run.
- [x] The saved checkpoint is `..._10_epochs.pth` while the notebook trains 20. Reconcile or retrain.

---

## Tier 1 — Evaluation protocol

Currently dev is folded into training, there's no validation set, and test accuracy is printed every epoch — test-set peeking for model selection.

**DAIC-WOZ is ~30% positive.** A constant "not depressed" predictor scores ~70–77% accuracy. Any accuracy in that band is indistinguishable from predicting nothing.

### 1.1 Fix the splits

- [x] Keep the official train (107) / dev (35) / test (47) split intact.
- [x] Select epochs, thresholds, and hyperparameters on **dev only**.
- [x] Touch test exactly once, at the end, and report that number unchanged.

### 1.2 Report metrics that mean something

- [x] **F1 (positive class)**, precision, recall, AUROC, AUPRC, confusion matrix.
- [x] Remove accuracy-only claims from the README.
- [x] Tune the decision threshold on dev (maximize F1 or fix recall at a chosen operating point) — `0.5` is arbitrary for an imbalanced task.

### 1.3 Baselines (non-negotiable)

| Baseline | Why |
|---|---|
| Majority class | Establishes the floor everything must clear |
| TF-IDF + logistic regression on transcripts | Frequently competitive on n=189; if you don't beat it, that's the finding |
| Text-only LSTM | Isolates whether audio/video contribute at all |
| Audio-only, video-only | Per-modality ablation for the results table |

### 1.4 Handle the small-sample problem

With 189 participants, a single split is noise.

- [x] Run **5 seeds minimum**; report mean ± std, not a single number.
- [ ] Consider nested cross-validation on train+dev for hyperparameter selection.
- [x] Report per-modality ablations with the same protocol.
- [ ] Break down errors by **gender** — a documented confound in this corpus.

---

## Tier 2 — Modeling upgrades

Ordered by expected payoff per unit effort.

### 2.1 Text — biggest win, least effort

- [ ] Replace static fastText with a fine-tuned transformer over participant turns: **RoBERTa**, or better, **MentalBERT / PsychBERT** (pretrained on mental-health corpora).
- [ ] Encode each turn independently, then aggregate turn embeddings with **attention pooling** rather than flattening 5,000 tokens into one LSTM.
- [ ] Add interpretable features alongside: first-person singular rate, absolutist-word rate, negative-emotion density, mean response latency, disfluency counts, turn-length distribution.
- [ ] Freeze the encoder and train only the head first; unfreeze the top layers later if dev F1 supports it.

### 2.2 Audio

100 Hz COVAREP frames over 5,000+ steps is hostile to an LSTM. Two viable directions:

- [ ] **Functionals route:** compute **eGeMAPS** (88 validated descriptors) per speech segment via openSMILE. Cheap, interpretable, strong baseline.
- [ ] **Self-supervised route:** if you can obtain raw audio, use **wav2vec 2.0 / WavLM / HuBERT** embeddings.
- [ ] Either way, segment by **participant turn** using transcript timestamps so you're not modeling the interviewer's speech or silence.

### 2.3 Video

- [ ] Replace CLNF with **OpenFace 2.0** (CLNF's successor).
- [ ] Use **normalized** AU intensities and head-pose dynamics — raw landmark pixel coordinates encode where the person sat relative to the camera, not their affect.
- [ ] Derive behavioural statistics rather than feeding 388 raw dims: AU12 (smile) rate, AU4 (brow lower) duration, gaze-aversion ratio, head-motion variance, blink rate.

### 2.4 Fusion

The README describes a "weighted approach" but the code does plain concatenation. Fix one or the other, then improve:

- [ ] **Cross-modal attention** — each modality attends over the others' turn-level representations.
- [ ] **Gated fusion** with learned per-modality weights (also gives you an interpretability signal).
- [ ] **Tensor Fusion Network / Low-rank Multimodal Fusion** as published baselines.
- [ ] **Modality dropout** during training so the model degrades gracefully when a modality is missing — the realistic deployment case.

### 2.5 Reformulate the task

- [ ] Predict the **PHQ-8 score** (regression: RMSE / MAE / **CCC**) alongside the binary label.
- [ ] Multi-task on the **eight individual item scores** — far more supervision from the same 189 subjects.
- [ ] CCC on PHQ-8 is what AVEC 2017 scored, so you get directly comparable published numbers.

### 2.6 Regularization

You're fitting ~250k parameters on 142 samples.

- [ ] Shrink `hidden_size` (try 16–32), add dropout and weight decay.
- [ ] Class-weighted `BCEWithLogitsLoss` (also numerically safer than `sigmoid` + `BCELoss`).
- [ ] Early stopping on **dev F1**, not dev loss.
- [ ] Gradient clipping for the recurrent layers.

---

## Tier 3 — Generalization

This is what separates a project from a homework assignment. Single-corpus results on DAIC-WOZ do not generalize — that's the open problem in the field.

- [ ] **Cross-corpus evaluation.** Train on DAIC-WOZ, test on **E-DAIC** (AVEC 2019), or AVEC 2013/2014 for the audio-visual side. Expect a large drop; reporting it honestly is worth more than another 2% in-domain.
- [ ] **Interpretability.** Attention weights over turns, SHAP over the engineered feature set, per-modality ablations. A clinician will never accept "87%" — they'll ask *which behaviours drove this*.
- [ ] **Calibration.** Reliability diagram and Brier score before you display any probability to a user.
- [ ] **Subgroup analysis.** Performance by gender and by PHQ-8 severity band.

---

## Tier 4 — Engineering and product

### 4.1 Reproducibility

- [x] `requirements.txt` with **pinned versions** (`torch==`, `gensim==`, `streamlit==`).
- [x] `LICENSE` file.
- [ ] `setup.sh` or `Makefile` that fetches large artifacts — the repo silently expects a 4.5 GB `crawl-300d-2M.vec` and a `models/` directory that doesn't exist.
- [x] `config.yaml` for all paths and hyperparameters; no hardcoded `./daic_woz/...` strings.
- [ ] Experiment tracking (Weights & Biases or MLflow) — you'll be running many seeds.
- [ ] Move all logic out of notebooks into `src/`; keep notebooks for exploration only.

### 4.2 Tests

- [ ] Shape assertions for each `return_tensor()`.
- [ ] Golden-file test: one fixed participant → known feature checksum.
- [ ] Train/inference parity test (see 0.1).
- [ ] A 2-sample overfit test — if the model can't reach ~100% on two examples, something is broken.

### 4.3 The app is the real bottleneck

**Asking users to upload seven pre-extracted COVAREP/CLNF files is a dead end.** Nobody outside the DAIC-WOZ archive has these files.

- [ ] Accept a **video or audio file**; run OpenFace + openSMILE + Whisper server-side to generate features.
- [ ] This single change turns a demo into something a person can actually use.
- [ ] Validate uploads: column counts, delimiters, empty files. The commented-out `all([...])` check suggests you already noticed this.
- [ ] Cache the word vectors with `@st.cache_resource` — currently every Streamlit rerun reloads 4.5 GB.
- [ ] Show per-modality confidence and which modalities were available.
- [ ] Add a processing progress indicator; feature extraction on a 15-minute video is slow.

---

## Tier 5 — Framing and ethics

> Please don't skip this section. It's also the cheapest tier to complete.

The app currently prints *"There is X% chance the person being depressed."* A raw sigmoid output is not a calibrated probability, and this is not a diagnosis.

- [ ] **Rename the project** from "diagnosis" to **screening support** or **risk indication**. This is also more accurate about what PHQ-8 labels represent — they're a self-report screening instrument, not a clinical diagnosis.
- [ ] **Calibrate** before displaying any percentage (Platt scaling or isotonic regression fit on dev). Report the calibration curve and Brier score.
- [ ] **Visible disclaimer** in the UI and README: research prototype, not a medical device, not for clinical use, not a substitute for professional assessment.
- [ ] **Signpost help.** If the tool outputs elevated risk, display appropriate mental-health support resources for the user's region.
- [ ] **Data-use statement.** DAIC-WOZ has access restrictions — document them and commit **no** participant data to the repo. Add `daic_woz/` to `.gitignore`.
- [ ] **Limitations section** in the README: n=189, single US sample, gender-imbalanced, Wizard-of-Oz interview setting that doesn't resemble clinical intake, unknown performance across cultures, dialects, and age groups.
- [ ] **Failure modes.** Be explicit that false negatives (missed cases) and false positives (unwarranted labelling) both carry real harm, and state which your threshold is tuned to favour and why.

---

## Suggested sequencing

### Weeks 1–2 — Foundation

Fix all Tier 0 bugs. Rebuild the pipeline as a proper `Dataset` / `DataLoader` with cached preprocessed tensors. Stand up the evaluation harness and all four baselines from 1.3.

> **Expect your real F1 to be much lower than the current reported numbers.** That's the point of this phase — you now have a trustworthy floor to improve from.

**Exit criteria:** honest dev F1 for majority-class, TF-IDF+LR, and the current LSTM, across 5 seeds, with the test set untouched.

### Weeks 3–5 — Modeling

Transformer text encoder, eGeMAPS or wav2vec audio, OpenFace video features. PHQ-8 regression as an auxiliary task. Attention fusion with modality dropout. Multi-seed results with dev-set-only selection.

**Exit criteria:** an ablation table (text / audio / video / pairs / all three) with mean ± std, beating the TF-IDF baseline on dev F1.

### Weeks 6–8 — Generalization and product

Cross-corpus evaluation on E-DAIC. Interpretability analysis. Calibration. Rebuild the app around raw media upload. Write it up.

**Exit criteria:** a clean README with a results table, ablations, cross-corpus numbers, and an honest limitations section.

> A clear, honest write-up is worth more than the model. If your fixed pipeline gives 0.60 F1 with a rigorous protocol, that's a good result you can defend — and defending it is the harder and more valuable skill.

---

## Target repo structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── config.yaml
├── Makefile
├── .gitignore                  # daic_woz/, models/, *.pth, *.vec
├── src/
│   ├── data/
│   │   ├── loaders.py          # official splits, ID-aligned
│   │   ├── dataset.py          # torch Dataset, caching
│   │   └── features/
│   │       ├── text.py         # SHARED by train + inference
│   │       ├── audio.py
│   │       └── video.py
│   ├── models/
│   │   ├── encoders.py
│   │   ├── fusion.py
│   │   └── baselines.py
│   ├── train.py
│   ├── evaluate.py             # F1/AUROC/AUPRC/CCC, calibration
│   └── calibrate.py
├── app/
│   ├── streamlit_app.py
│   └── extract.py              # OpenFace / openSMILE / Whisper
├── tests/
│   ├── test_shapes.py
│   ├── test_train_infer_parity.py
│   └── test_overfit_two_samples.py
├── notebooks/                  # exploration only
└── results/
    ├── ablations.md
    └── figures/
```

---

## Reference numbers

Keep these in view so you can sanity-check your own results.

| Quantity | Value |
|---|---|
| DAIC-WOZ participants | 189 (train 107 / dev 35 / test 47) |
| Positive rate | ~30% |
| Constant-predictor accuracy | ~70–77% |
| Typical published F1, depressed class | ~0.55–0.75 |
| COVAREP | 74 features @ 100 Hz |
| CLNF video | 30 fps |
| Current model parameters | ~250k, fit on 142 samples |

Many papers reporting 90%+ accuracy on this dataset have leakage, an undisclosed split change, or are reporting accuracy on an imbalanced set. Treat any such number — including your own — with suspicion until you can trace the protocol.

---

## Quick-reference checklist

**Blocking (Tier 0)** — done, see [results/ablations.md](results/ablations.md)

- [x] Unify text preprocessing between training and inference
- [x] Build video features for dev and test splits
- [x] Fix `scale_down()` in-place mutation and indexing
- [x] Drop timestamp; normalize all features with train-only statistics
- [x] Align modality time windows (uniform full-interview resampling, not transcript-timestamp segment alignment — see 0.5)
- [x] Remove stopword filtering
- [x] Raise on unmatched participant IDs
- [x] `weights_only=True`; unify `torch.cat` dim; set seeds

**High-value (Tiers 1–2)**

- [x] Restore dev set; select on dev; touch test once
- [x] Report F1 / AUROC / AUPRC + confusion matrix
- [x] Add four baselines
- [x] 5-seed mean ± std
- [ ] Transformer text encoder
- [ ] eGeMAPS or wav2vec audio
- [ ] OpenFace normalized video features
- [ ] PHQ-8 regression auxiliary task
- [ ] Attention fusion + modality dropout

**Credibility (Tiers 3–5)**

- [ ] Cross-corpus eval on E-DAIC
- [ ] Interpretability + calibration
- [x] `requirements.txt`, `LICENSE`, tests, config
- [ ] App accepts raw media
- [ ] Rename to "screening"; add disclaimers, limitations, data-use statement (README/app disclaimer added; full ethics tier not done)
