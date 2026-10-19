"""
Unified evaluation protocol.

Every model is scored with exactly these metrics, on the same test set, with
the operating threshold chosen the same way (Youden's J on VALIDATION, never
on test) and with probabilities temperature-calibrated on VALIDATION.

Reported metrics (paper requirement):
    Accuracy, Balanced Accuracy, Precision, Recall (sensitivity), F1
plus AUC-ROC, PR-AUC, specificity, ECE and the confusion matrix.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve)


# ---------------------------------------------------------------------------
#  Threshold selection -- fitted on VALIDATION only
# ---------------------------------------------------------------------------
def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, rule: str = "youden") -> float:
    if rule == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        return float(thr[int(np.argmax(tpr - fpr))])
    if rule == "f1":
        grid = np.linspace(0.05, 0.95, 181)
        scores = [f1_score(y_true, (y_prob >= t).astype(int), average="macro") for t in grid]
        return float(grid[int(np.argmax(scores))])
    if rule == "half":
        return 0.5
    raise ValueError(f"unknown threshold rule {rule!r}")


# ---------------------------------------------------------------------------
#  Temperature scaling -- fitted on VALIDATION only
# ---------------------------------------------------------------------------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        z = torch.tensor(logits, dtype=torch.float32).view(-1)
        y = torch.tensor(labels, dtype=torch.float32).view(-1)
        opt = torch.optim.LBFGS([self.log_t], lr=0.05, max_iter=100)
        lossf = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = lossf(self.forward(z), y)
            loss.backward()
            return loss

        opt.step(closure)
        return self.temperature


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------
#  The metric block
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision":         float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":      float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity":       float(specificity),
        "auc":               float(roc_auc_score(y_true, y_prob)),
        "pr_auc":            float(average_precision_score(y_true, y_prob)),
        "ece":               expected_calibration_error(y_true, y_prob),
        "threshold":         float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                 n: int = 1000, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """Percentile 95% CIs, so the paper can state whether gaps are meaningful."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]
    acc: Dict[str, list] = {k: [] for k in keys}
    idx_all = np.arange(len(y_true))
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], y_prob[idx], threshold)
        for k in keys:
            acc[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in acc.items() if v}


def delong_like_pvalue(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                       n: int = 2000, seed: int = 42) -> float:
    """
    Bootstrap test for AUC(A) > AUC(B) on paired predictions.
    Used to state whether the top model is genuinely ahead of the runner-up.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int).ravel()
    diffs = []
    idx_all = np.arange(len(y_true))
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], p_a[idx]) - roc_auc_score(y_true[idx], p_b[idx]))
    diffs = np.asarray(diffs)
    return float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
