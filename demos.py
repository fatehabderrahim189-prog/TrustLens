"""Self-contained demo audits on public/synthetic *proxy* data.

These are stand-ins that show the workflow; plug in your real models (see README)."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

from .report import audit_classifier, audit_regressor


def demo_radioscreen_proxy(seed=0):
    d = load_breast_cancer()
    X, y = d.data, d.target
    Xtr, Xr, ytr, yr = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    Xc, Xt, yc, yt = train_test_split(Xr, yr, test_size=0.6, stratify=yr, random_state=seed)
    model = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=seed).fit(Xtr, ytr)
    groups = np.where(Xt[:, 3] > np.median(X[:, 3]), "larger lesion area", "smaller lesion area")
    return audit_classifier("RadioScreen (proxy: tabular diagnostic data)", model.predict_proba, Xc, yc, Xt, yt,
                            X_ref=Xtr, feature_names=list(d.feature_names), groups_test=groups, seed=seed,
                            description="Stand-in for AI-assisted chest X-ray screening, using the public Wisconsin diagnostic dataset. Replace with RadioScreen's real probabilities.")


def demo_aegis_proxy(seed=0):
    rng = np.random.default_rng(seed)
    X, y = make_classification(6000, 12, n_informative=6, weights=[0.93, 0.07], class_sep=1.0, flip_y=0.02, random_state=seed)
    Xtr, Xr, ytr, yr = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    Xc, Xt, yc, yt = train_test_split(Xr, yr, test_size=0.5, stratify=yr, random_state=seed)
    model = RandomForestClassifier(200, min_samples_leaf=3, random_state=seed).fit(Xtr, ytr)
    Xt = Xt.copy(); Xt[:, 0] += 0.9 * X[:, 0].std(); Xt[:, 3] *= 1.35          # simulated traffic-pattern drift
    groups = rng.choice(["sensor", "PLC", "gateway"], size=len(yt))
    return audit_classifier("AEGIS-Net (proxy: imbalanced intrusion detection under drift)", model.predict_proba, Xc, yc, Xt, yt,
                            X_ref=Xtr, feature_names=[f"flow_feat_{i}" for i in range(12)], groups_test=groups, seed=seed,
                            description="Stand-in for federated IIoT intrusion detection: 7% attack rate and a simulated traffic-pattern shift at test time.")


def demo_axiom_zero_proxy(seed=0):
    rng = np.random.default_rng(seed)
    n = 360; t = np.arange(n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 0.6)
    y = 3.0 + 1.2 * np.sin(2 * np.pi * t / 12) + ar                                   # stationary KPI (e.g. growth rate, %)
    s0 = int(n * 0.8)
    y[s0:] += 1.2 + rng.normal(0, 0.5, n - s0)                                         # regime shift: higher mean, more volatility
    lag = lambda k: np.r_[np.full(k, y[0]), y[:-k]]
    X = np.column_stack([lag(1), lag(2), lag(12), np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12)])[12:]
    yy = y[12:]
    a, b = int(len(yy) * 0.6), int(len(yy) * 0.8)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed).fit(X[:a], yy[:a])
    return audit_regressor("Axiom-Zero (proxy: economic KPI forecasting with regime shift)", model.predict,
                           X[a:b], yy[a:b], X[b:], yy[b:], X_ref=X[:a],
                           feature_names=["kpi_lag1", "kpi_lag2", "kpi_lag12", "season_sin", "season_cos"], seed=seed,
                           description="Stand-in for KPI forecasting: time-ordered split with a structural break (mean and volatility) in the test period.")


ALL = {"radioscreen": demo_radioscreen_proxy, "aegis": demo_aegis_proxy, "axiom": demo_axiom_zero_proxy}
