"""Calibration metrics and post-hoc temperature scaling (Guo et al., 2017)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def as_2d(proba) -> np.ndarray:
    """Accept binary (n,) positive-class probabilities or (n, k) matrices."""
    p = np.asarray(proba, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    return np.clip(p, 1e-12, 1.0)


def confidence_and_correct(y_true, proba):
    p = as_2d(proba)
    y = np.asarray(y_true, dtype=int)
    return p.max(1), p.argmax(1) == y


def reliability_curve(y_true, proba, n_bins: int = 10) -> dict:
    conf, correct = confidence_and_correct(y_true, proba)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    b_conf, b_acc, b_n = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            b_conf.append(conf[m].mean()); b_acc.append(correct[m].mean()); b_n.append(int(m.sum()))
    return {"confidence": np.array(b_conf), "accuracy": np.array(b_acc), "count": np.array(b_n)}


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    r = reliability_curve(y_true, proba, n_bins)
    n = r["count"].sum()
    return float(np.sum(r["count"] / n * np.abs(r["accuracy"] - r["confidence"])))


def max_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    r = reliability_curve(y_true, proba, n_bins)
    return float(np.max(np.abs(r["accuracy"] - r["confidence"])))


def brier_score(y_true, proba) -> float:
    p = as_2d(proba)
    onehot = np.eye(p.shape[1])[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def log_loss(y_true, proba) -> float:
    p = as_2d(proba)
    y = np.asarray(y_true, dtype=int)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_temperature(proba, temperature: float) -> np.ndarray:
    """Rescale probabilities (treated as log-logits) by a temperature. Returns (n, k)."""
    return _softmax(np.log(as_2d(proba)) / temperature)


def fit_temperature(y_true, proba) -> float:
    """Fit the single temperature T minimising NLL on a held-out calibration set."""
    p = as_2d(proba)
    y = np.asarray(y_true, dtype=int)
    logits = np.log(p)

    def nll(t):
        q = _softmax(logits / t)
        return -np.mean(np.log(q[np.arange(len(y)), y] + 1e-12))

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
