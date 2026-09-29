"""Model-agnostic explanation and subgroup analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd


def permutation_importance(predict_fn, X, y, metric_fn, feature_names=None,
                           n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Drop in `metric_fn(y, predict_fn(X))` (higher = better) when one feature is shuffled."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    base = metric_fn(y, predict_fn(X))
    names = feature_names or [f"x{i}" for i in range(X.shape[1])]
    rows = []
    for j, nme in enumerate(names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - metric_fn(y, predict_fn(Xp)))
        rows.append({"feature": nme, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def subgroup_report(y_true, y_pred, groups, min_n: int = 20) -> tuple[pd.DataFrame, float]:
    """Per-group accuracy and the largest gap between groups with at least `min_n` samples."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "g": np.asarray(groups)})
    out = (df.assign(ok=df.y == df.p).groupby("g")
           .agg(n=("ok", "size"), accuracy=("ok", "mean")).reset_index().rename(columns={"g": "group"}))
    valid = out[out.n >= min_n]
    gap = float(valid.accuracy.max() - valid.accuracy.min()) if len(valid) > 1 else 0.0
    return out, gap
