"""Evaluation harness (Tier 1.2): F1/precision/recall/AUROC/AUPRC/confusion
matrix, with the decision threshold tuned on dev rather than fixed at 0.5.

DAIC-WOZ is ~30% positive: a constant "not depressed" predictor scores
~70-77% accuracy, so accuracy alone is not reported here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold in [0.05, 0.95] that maximizes F1 on the given (dev) set."""
    candidates = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": float((y_pred == y_true).mean()),
        "n": len(y_true),
        "n_positive": int(y_true.sum()),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()  # [[tn, fp], [fn, tp]]
    return metrics


def summarize_seeds(per_seed_metrics: list[dict]) -> dict:
    """Mean +/- std across seeds (Tier 1.4: 5 seeds minimum)."""
    keys = ["f1", "precision", "recall", "accuracy", "auroc", "auprc"]
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics], dtype=float)
        summary[f"{k}_mean"] = float(np.nanmean(vals))
        summary[f"{k}_std"] = float(np.nanstd(vals))
    summary["n_seeds"] = len(per_seed_metrics)
    return summary


def format_metrics(name: str, metrics: dict) -> str:
    return (
        f"{name}: F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"AUROC={metrics['auroc']:.3f} AUPRC={metrics['auprc']:.3f} "
        f"(thr={metrics['threshold']:.2f}, n={metrics['n']}, pos={metrics['n_positive']})"
    )
