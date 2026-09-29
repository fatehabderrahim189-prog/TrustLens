"""Audit orchestration, Trust Score, HTML dashboard and Model Card generation."""
from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .calibration import (apply_temperature, as_2d, brier_score, confidence_and_correct,  # noqa: E402
                          expected_calibration_error, fit_temperature, log_loss,
                          max_calibration_error, reliability_curve)
from .drift import drift_report  # noqa: E402
from .explain import permutation_importance, subgroup_report  # noqa: E402
from .uncertainty import (bootstrap_ci, conformal_interval_width, conformal_threshold,  # noqa: E402
                          prediction_sets, risk_coverage_curve, selective_accuracy,
                          set_coverage, threshold_for_risk)

BG, FG, ACC, WARN, GRID = "#0b1220", "#dbe7ff", "#4da3ff", "#ff7a59", "#22314d"

VERDICTS = [(80, "Deployable with monitoring", "قابل للنشر مع مراقبة", "#3ddc97"),
            (60, "Human-in-the-loop required", "يتطلب إشرافاً بشرياً", "#ffc857"),
            (0, "Not ready for autonomous use", "غير جاهز للاستخدام المستقل", "#ff7a59")]


def _clip100(x: float) -> float:
    return float(np.clip(x, 0.0, 100.0))


@dataclass
class AuditResult:
    name: str
    task: str
    description: str
    n_cal: int
    n_test: int
    alpha: float
    metrics: dict = field(default_factory=dict)
    pillars: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    figures: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    oversight: dict = field(default_factory=dict)

    @property
    def trust_score(self) -> float:
        return float(np.mean(list(self.pillars.values()))) if self.pillars else 0.0

    @property
    def verdict(self):
        s = self.trust_score
        for thr, en, ar, col in VERDICTS:
            if s >= thr:
                return en, ar, col

    def to_dict(self) -> dict:
        return {"name": self.name, "task": self.task, "trust_score": round(self.trust_score, 1),
                "verdict": self.verdict[0], "pillars": {k: round(v, 1) for k, v in self.pillars.items()},
                "metrics": self.metrics, "oversight": self.oversight, "notes": self.notes,
                "n_cal": self.n_cal, "n_test": self.n_test, "alpha": self.alpha}


# ------------------------------------------------------------------ figures
def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _ax(w=5.2, h=3.6, title=""):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=FG, fontsize=10)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    return fig, ax


def _fig_reliability(y, p, n_bins):
    r = reliability_curve(y, p, n_bins)
    fig, ax = _ax(title="Reliability diagram")
    ax.plot([0, 1], [0, 1], "--", color=GRID, label="perfect")
    ax.plot(r["confidence"], r["accuracy"], "o-", color=ACC, label="model")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend(facecolor=BG, labelcolor=FG, fontsize=8)
    return _b64(fig)


def _fig_risk_cov(conf, correct, thr):
    cov, risk = risk_coverage_curve(conf, correct)
    fig, ax = _ax(title="Risk-coverage (auto-handle vs escalate)")
    ax.plot(cov, risk, color=ACC)
    if thr["coverage"] > 0:
        ax.plot(thr["coverage"], thr["risk"], "o", color=WARN)
    ax.set_xlabel("coverage (share handled automatically)"); ax.set_ylabel("error rate")
    return _b64(fig)


def _fig_bars(df, col, title, label_col="feature", refs=()):
    d = df.head(10).iloc[::-1]
    fig, ax = _ax(title=title)
    ax.barh(d[label_col], d[col], color=ACC)
    for r in refs:
        ax.axvline(r, color=WARN, ls="--", lw=0.8)
    return _b64(fig)


def _fig_regression(y, pred, lo, hi):
    fig, ax = _ax(6.4, 3.6, title="Test period: actual vs prediction with conformal band")
    x = np.arange(len(y))
    ax.fill_between(x, lo, hi, color=ACC, alpha=0.2, label="conformal interval")
    ax.plot(x, y, color=FG, lw=1, label="actual")
    ax.plot(x, pred, color=WARN, lw=1, label="prediction")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8)
    return _b64(fig)


# ------------------------------------------------------------------ audits
def _fairness_and_drift(res, X_ref, X_test, feature_names, pillars):
    if X_ref is not None:
        d = drift_report(X_ref, X_test, feature_names)
        res.tables["drift"] = d
        res.figures["drift"] = _fig_bars(d, "psi_adj", "Feature drift (noise-adjusted PSI)", refs=(0.1, 0.25))
        mx = float(d.psi_adj.max())
        res.metrics["max_psi_adj"] = mx
        pillars["Stability"] = _clip100(100 * (1 - mx / 0.25))
        if mx >= 0.25:
            res.notes.append(f"Major covariate shift on '{d.feature[0]}' (adjusted PSI={mx:.2f}); expect degraded performance and re-validate before use.")


def audit_classifier(name, predict_proba, X_cal, y_cal, X_test, y_test, *, X_ref=None,
                     feature_names=None, groups_test=None, alpha=0.1, target_risk=0.05,
                     n_bins=10, seed=0, description="") -> AuditResult:
    X_cal, X_test = np.asarray(X_cal, float), np.asarray(X_test, float)
    y_cal, y_test = np.asarray(y_cal).astype(int), np.asarray(y_test).astype(int)
    p_cal, p_test = as_2d(predict_proba(X_cal)), as_2d(predict_proba(X_test))
    res = AuditResult(name, "classification", description, len(y_cal), len(y_test), alpha)
    m, pillars = res.metrics, {}

    acc = bootstrap_ci(lambda y, p: float((p.argmax(1) == y).mean()), y_test, p_test, seed=seed)
    ece = bootstrap_ci(lambda y, p: expected_calibration_error(y, p, n_bins), y_test, p_test, seed=seed)
    m.update(accuracy=acc[0], accuracy_ci=[acc[1], acc[2]], ece=ece[0], ece_ci=[ece[1], ece[2]],
             mce=max_calibration_error(y_test, p_test, n_bins), brier=brier_score(y_test, p_test),
             log_loss=log_loss(y_test, p_test))
    T = fit_temperature(y_cal, p_cal)
    ece_T = expected_calibration_error(y_test, apply_temperature(p_test, T), n_bins)
    m.update(temperature=T, ece_after_temperature_scaling=ece_T)
    pillars["Calibration"] = _clip100(100 * (1 - m["ece"] / 0.15))
    res.figures["reliability"] = _fig_reliability(y_test, p_test, n_bins)
    if m["ece"] > 0.05:
        res.notes.append(f"Confidence is miscalibrated (ECE={m['ece']:.3f}). Temperature scaling (T={T:.2f}) on the calibration set gives ECE={ece_T:.3f}.")

    q = conformal_threshold(p_cal, y_cal, alpha)
    sets = prediction_sets(p_test, q)
    cov = set_coverage(y_test, sets)
    m.update(conformal_target=1 - alpha, conformal_coverage=cov, mean_set_size=float(sets.sum(1).mean()),
             singleton_rate=float((sets.sum(1) == 1).mean()))
    pillars["Uncertainty coverage"] = _clip100(100 * (1 - abs(cov - (1 - alpha)) / 0.10))
    if cov < 1 - alpha - 0.03:
        res.notes.append(f"Conformal coverage {cov:.1%} is below the {1-alpha:.0%} target: exchangeability between calibration and test data is violated (shift).")

    conf, correct = confidence_and_correct(y_test, p_test)
    cv, rk = risk_coverage_curve(conf, correct)
    acc80 = selective_accuracy(conf, correct, 0.8)
    thr = threshold_for_risk(conf, correct, target_risk)
    m.update(selective_accuracy_at_80_coverage=acc80, aurc=float(rk.mean()))
    risk_full = 1 - m["accuracy"]
    pillars["Selective safety"] = 100.0 if risk_full < 1e-9 else _clip100(100 * (risk_full - (1 - acc80)) / risk_full)
    res.oversight = {"target_error_rate": target_risk, **thr}
    res.figures["risk_coverage"] = _fig_risk_cov(conf, correct, thr)

    imp = permutation_importance(lambda X: as_2d(predict_proba(X)).argmax(1), X_test, y_test,
                                 lambda y, p: float((y == p).mean()), feature_names, seed=seed)
    res.tables["importance"] = imp
    res.figures["importance"] = _fig_bars(imp, "importance", "Permutation importance (accuracy drop)")

    _fairness_and_drift(res, X_ref, X_test, feature_names, pillars)
    if groups_test is not None:
        sg, gap = subgroup_report(y_test, p_test.argmax(1), groups_test)
        res.tables["subgroups"], m["subgroup_gap"] = sg, gap
        pillars["Subgroup parity"] = _clip100(100 * (1 - gap / 0.15))
        if gap > 0.05:
            res.notes.append(f"Accuracy differs by {gap:.1%} across subgroups; review before deployment.")
    res.pillars = pillars
    return res


def audit_regressor(name, predict, X_cal, y_cal, X_test, y_test, *, X_ref=None, feature_names=None,
                    alpha=0.1, seed=0, description="") -> AuditResult:
    X_cal, X_test = np.asarray(X_cal, float), np.asarray(X_test, float)
    y_cal, y_test = np.asarray(y_cal, float), np.asarray(y_test, float)
    pc, pt = np.asarray(predict(X_cal), float), np.asarray(predict(X_test), float)
    res = AuditResult(name, "regression", description, len(y_cal), len(y_test), alpha)
    m, pillars = res.metrics, {}

    mae = bootstrap_ci(lambda y, p: float(np.mean(np.abs(y - p))), y_test, pt, seed=seed)
    ss_res, ss_tot = np.sum((y_test - pt) ** 2), np.sum((y_test - y_test.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    mae_cal = float(np.mean(np.abs(y_cal - pc)))
    m.update(mae=mae[0], mae_ci=[mae[1], mae[2]], rmse=float(np.sqrt(np.mean((y_test - pt) ** 2))),
             r2=r2, mae_calibration_set=mae_cal, degradation_ratio=mae[0] / mae_cal)
    pillars["Predictive skill"] = _clip100(100 * r2)
    pillars["Generalisation"] = _clip100(100 * (1 - (m["degradation_ratio"] - 1.0)))
    if m["degradation_ratio"] > 1.5:
        res.notes.append(f"Test error is {m['degradation_ratio']:.1f}x the calibration error: performance degrades out-of-sample.")

    w = conformal_interval_width(pc, y_cal, alpha)
    lo, hi = pt - w, pt + w
    cov = float(np.mean((y_test >= lo) & (y_test <= hi)))
    m.update(conformal_target=1 - alpha, conformal_coverage=cov, interval_half_width=w)
    pillars["Uncertainty coverage"] = _clip100(100 * (1 - abs(cov - (1 - alpha)) / 0.10))
    if cov < 1 - alpha - 0.03:
        res.notes.append(f"Interval coverage {cov:.1%} misses the {1-alpha:.0%} target: the error distribution shifted after calibration.")
    res.figures["forecast"] = _fig_regression(y_test, pt, lo, hi)

    imp = permutation_importance(predict, X_test, y_test,
                                 lambda y, p: float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
                                 feature_names, seed=seed)
    res.tables["importance"] = imp
    res.figures["importance"] = _fig_bars(imp, "importance", "Permutation importance (R² drop)")
    _fairness_and_drift(res, X_ref, X_test, feature_names, pillars)
    res.pillars = pillars
    return res


# ------------------------------------------------------------------ rendering
_CSS = """*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#dbe7ff;font:15px/1.55 system-ui,Segoe UI,Roboto,sans-serif}
.w{max-width:1000px;margin:auto;padding:24px 16px}h1{margin:0;font-size:26px}h2{font-size:17px;margin:28px 0 10px;color:#4da3ff}
.sub{color:#8fa6cc}.card{background:#111b30;border:1px solid #22314d;border-radius:14px;padding:16px;margin:12px 0}
.hero{display:flex;gap:22px;align-items:center;flex-wrap:wrap}.g{width:130px;height:130px;border-radius:50%;display:grid;place-items:center;
background:conic-gradient(var(--c) calc(var(--s)*1%),#22314d 0)}.g div{width:100px;height:100px;border-radius:50%;background:#111b30;display:grid;place-items:center;font-size:30px;font-weight:700}
.b{height:8px;background:#22314d;border-radius:5px;overflow:hidden}.b i{display:block;height:100%;background:#4da3ff}.row{margin:9px 0}.row span{display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}img{max-width:100%;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:6px 8px;border-bottom:1px solid #22314d;text-align:left}.tag{display:inline-block;padding:3px 10px;border-radius:99px;font-weight:600;color:#0b1220}
li{margin:5px 0}.ov{border-left:4px solid #4da3ff}.scroll{overflow-x:auto}a{color:#4da3ff}"""


def _tbl(df: pd.DataFrame) -> str:
    return '<div class="scroll">' + df.to_html(index=False, float_format=lambda v: f"{v:.3f}", border=0) + "</div>"


def render_html(res: AuditResult) -> str:
    en, ar, col = res.verdict
    e = html.escape
    pill = "".join(f'<div class="row"><span><b>{e(k)}</b><b>{v:.0f}/100</b></span><div class="b"><i style="width:{v:.0f}%"></i></div></div>'
                   for k, v in res.pillars.items())
    figs = "".join(f'<div class="card"><img alt="{e(k)}" src="data:image/png;base64,{v}"></div>' for k, v in res.figures.items())
    tabs = "".join(f"<h2>{e(k.title())}</h2><div class='card'>{_tbl(t)}</div>" for k, t in res.tables.items())
    notes = "".join(f"<li>{e(n)}</li>" for n in res.notes) or "<li>No critical issues detected by the automatic checks.</li>"
    m = res.metrics
    mrows = "".join(f"<tr><td>{e(k)}</td><td>{e(', '.join(f'{x:.4f}' for x in v) if isinstance(v, list) else (f'{v:.4f}' if isinstance(v, float) else str(v)))}</td></tr>" for k, v in m.items())
    ov = ""
    if res.oversight:
        o = res.oversight
        ov = (f'<div class="card ov"><b>Human-oversight policy.</b> To keep the error rate of auto-handled cases ≤ {o["target_error_rate"]:.0%}, '
              f'escalate every case with confidence below <b>{o["threshold"]:.3f}</b> to a human reviewer — about '
              f'<b>{o["review_rate"]:.1%}</b> of cases.</div>' if np.isfinite(o["threshold"]) else
              '<div class="card ov"><b>Human-oversight policy.</b> No confidence threshold reaches the target error rate: route all cases to human review.</div>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrustLens · {e(res.name)}</title><style>{_CSS}</style></head><body><div class="w">
<div class="sub">TrustLens audit · {e(res.task)} · calibration n={res.n_cal}, test n={res.n_test}</div><h1>{e(res.name)}</h1><p class="sub">{e(res.description)}</p>
<div class="card hero"><div class="g" style="--s:{res.trust_score:.0f};--c:{col}"><div>{res.trust_score:.0f}</div></div>
<div><span class="tag" style="background:{col}">{e(en)}</span><p dir="rtl" style="margin:8px 0 0">{e(ar)}</p>
<p class="sub" style="margin:8px 0 0">Trust Score = mean of the pillar scores below (transparent heuristic, not a certification).</p></div></div>
<h2>Pillars</h2><div class="card">{pill}</div>{ov}<h2>Findings</h2><div class="card"><ul>{notes}</ul></div>
<h2>Diagnostics</h2><div class="grid">{figs}</div><h2>Metrics</h2><div class="card scroll"><table>{mrows}</table></div>{tabs}
<p class="sub">Generated by TrustLens. Scores depend on the quality and representativeness of the calibration/test data supplied.</p></div></body></html>"""


def model_card_md(res: AuditResult) -> str:
    m = res.metrics
    en, _, _ = res.verdict
    lines = [f"# Model Card — {res.name}", "", f"*Auto-generated by TrustLens · task: {res.task}*", "",
             "## Summary", res.description or "_No description supplied._", "",
             f"**Trust Score: {res.trust_score:.0f}/100 — {en}**", "", "| Pillar | Score |", "|---|---|"]
    lines += [f"| {k} | {v:.0f} |" for k, v in res.pillars.items()]
    lines += ["", "## Key metrics", "", "| Metric | Value |", "|---|---|"]
    for k, v in m.items():
        s = ", ".join(f"{x:.4f}" for x in v) if isinstance(v, list) else (f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(f"| {k} | {s} |")
    if res.oversight:
        o = res.oversight
        lines += ["", "## Human oversight", ""]
        if np.isfinite(o["threshold"]):
            lines.append(f"Escalate cases with confidence < {o['threshold']:.3f} (≈{o['review_rate']:.1%} of cases) to keep auto-handled error ≤ {o['target_error_rate']:.0%}.")
        else:
            lines.append("No threshold meets the target error rate; all cases require human review.")
    lines += ["", "## Findings and limitations"] + [f"- {n}" for n in (res.notes or ["No critical issues detected by automatic checks."])]
    lines += ["- Trust Score is a transparent heuristic, not a safety certification.",
              "- Results are valid only for data resembling the calibration/test sets used here.", ""]
    return "\n".join(lines)
