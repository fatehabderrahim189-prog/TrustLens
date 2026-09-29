"""Uncertainty quantification: bootstrap CIs, split-conformal prediction, selective prediction."""
from __future__ import annotations

import math

import numpy as np

from .calibration import as_2d


def bootstrap_ci(metric_fn, *arrays, n_boot: int = 300, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI. Returns (point_estimate, low, high)."""
    arrays = [np.asarray(a) for a in arrays]
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(*[a[idx] for a in arrays]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(metric_fn(*arrays)), float(lo), float(hi)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


# ---- classification: split conformal (Vovk; Angelopoulos & Bates) -------------------------
def conformal_threshold(cal_proba, cal_y, alpha: float = 0.1) -> float:
    p = as_2d(cal_proba)
    y = np.asarray(cal_y, dtype=int)
    return _conformal_quantile(1.0 - p[np.arange(len(y)), y], alpha)


def prediction_sets(proba, qhat: float) -> np.ndarray:
    """Boolean (n, k) mask; a class is in the set when its probability >= 1 - qhat."""
    return as_2d(proba) >= (1.0 - qhat)


def set_coverage(y_true, sets: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    return float(sets[np.arange(len(y)), y].mean())


# ---- regression: split conformal on absolute residuals ------------------------------------
def conformal_interval_width(cal_pred, cal_y, alpha: float = 0.1) -> float:
    return _conformal_quantile(np.abs(np.asarray(cal_y) - np.asarray(cal_pred)), alpha)


# ---- selective prediction: abstain / escalate to a human -------------------------------
def risk_coverage_curve(conf, correct):
    """Sort by confidence (desc). Returns (coverage, risk) where risk = error rate at that coverage."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    err = 1.0 - np.asarray(correct, dtype=float)[order]
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def selective_accuracy(conf, correct, coverage: float = 0.8) -> float:
    n = len(conf)
    k = max(1, int(math.ceil(coverage * n)))
    order = np.argsort(-np.asarray(conf), kind="stable")[:k]
    return float(np.asarray(correct, dtype=float)[order].mean())


def threshold_for_risk(conf, correct, target_risk: float = 0.05) -> dict:
    """Lowest confidence threshold whose auto-handled cases keep error <= target_risk.
    Cases below the threshold should be escalated to a human reviewer."""
    conf = np.asarray(conf)
    cov, risk = risk_coverage_curve(conf, correct)
    ok = np.where(risk <= target_risk)[0]
    if len(ok) == 0:
        return {"threshold": float("inf"), "coverage": 0.0, "risk": float("nan"), "review_rate": 1.0}
    k = ok[-1]
    thr = float(np.sort(conf)[::-1][k])
    return {"threshold": thr, "coverage": float(cov[k]), "risk": float(risk[k]),
            "review_rate": float(1.0 - cov[k])}
