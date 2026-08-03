"""Non-negotiable baselines (Tier 1.3):
- majority class: the floor every model must clear
- TF-IDF + logistic regression on transcripts
- (unimodal LSTMs live in train.py, which reuses UnimodalLSTM from fusion.py)
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..evaluate import best_threshold, compute_metrics


def majority_class_baseline(train_labels: list[int], eval_labels: list[int]) -> dict:
    majority = int(round(np.mean(train_labels)))
    y_prob = np.full(len(eval_labels), float(majority))
    return compute_metrics(eval_labels, y_prob, threshold=0.5)


def tfidf_logreg_baseline(train_texts: list[str], train_labels: list[int],
                           dev_texts: list[str], dev_labels: list[int],
                           test_texts: list[str], test_labels: list[int], seed: int = 0) -> dict:
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_texts)
    X_dev = vectorizer.transform(dev_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(X_train, train_labels)

    dev_prob = clf.predict_proba(X_dev)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

    threshold = best_threshold(np.array(dev_labels), dev_prob)
    dev_metrics = compute_metrics(dev_labels, dev_prob, threshold=threshold)
    test_metrics = compute_metrics(test_labels, test_prob, threshold=threshold)
    return {"dev": dev_metrics, "test": test_metrics}


def joined_turns(turns: list[str]) -> str:
    return " ".join(turns)
