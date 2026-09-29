"""Distribution-shift detection: Population Stability Index (noise-adjusted) and two-sample KS test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected, actual, bins: int = 10) -> float:
    """PSI with quantile bins of the reference sample. <0.1 stable, 0.1-0.25 moderate, >0.25 major."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.clip(np.histogram(e, edges)[0] / len(e), 1e-4, None)
    pa = np.clip(np.histogram(a, edges)[0] / len(a), 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_noise_floor(n_ref: int, n_new: int, bins: int = 10) -> float:
    """Expected PSI between two samples of the SAME distribution (finite-sample bias)."""
    return (bins - 1) * (1.0 / n_ref + 1.0 / n_new)


def _status(v: float) -> str:
    return "stable" if v < 0.1 else "moderate" if v < 0.25 else "major"


def drift_report(X_ref, X_new, feature_names=None) -> pd.DataFrame:
    X_ref, X_new = np.asarray(X_ref, dtype=float), np.asarray(X_new, dtype=float)
    names = feature_names or [f"x{i}" for i in range(X_ref.shape[1])]
    rows = []
    for j, nme in enumerate(names):
        stat, p = ks_2samp(X_ref[:, j], X_new[:, j])
        v = psi(X_ref[:, j], X_new[:, j])
        adj = max(0.0, v - psi_noise_floor(len(X_ref), len(X_new)))
        rows.append({"feature": nme, "psi": v, "psi_adj": adj, "ks_stat": float(stat), "ks_p": float(p), "status": _status(adj)})
    return pd.DataFrame(rows).sort_values("psi_adj", ascending=False).reset_index(drop=True)
