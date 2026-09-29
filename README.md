# TrustLens 🔍

**A model-agnostic trust & evaluation layer for human-governed AI.**
Give it any classifier or regressor (just a `predict` function) and a held-out calibration/test set. It returns an auditable report: is the model's confidence honest, how uncertain is it, **which cases must a human review**, has the data shifted, and what drives the predictions.

[![CI](https://github.com/fatehabderrahim189-prog/TrustLens/actions/workflows/ci.yml/badge.svg)](https://github.com/fatehabderrahim189-prog/TrustLens/actions)
![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

> Built as the shared evaluation layer for my projects **AEGIS-Net** (IIoT security), **RadioScreen** (chest X-ray screening), **Axiom-Zero** (economic KPI forecasting) and the **Sovereign AI Workforce** governance work: *research • build • evaluate • explain • improve*.

## What it does

| Pillar | Method | Question answered |
|---|---|---|
| Calibration | ECE / MCE / Brier / NLL + temperature scaling | Does "90% confident" mean right 90% of the time? |
| Uncertainty coverage | Split-conformal prediction sets (classification) and intervals (regression) | Do the sets/intervals really cover the truth at the promised rate? |
| Selective safety | Risk–coverage curve + escalation threshold | Which cases should go to a human to keep auto-handled error ≤ a target? |
| Stability | Sample-size-adjusted PSI + KS test per feature | Has the input distribution moved since training? |
| Subgroup parity | Per-group accuracy gap | Does performance differ across groups? |
| Explanation | Model-agnostic permutation importance | What is the model relying on? |

Outputs: a **self-contained HTML dashboard**, an auto-generated **Model Card (Markdown)**, and a machine-readable `summary.json`. The **Trust Score** (0–100) is the mean of the pillar scores. Formulas are in `trustlens/report.py` and are a transparent heuristic, **not a safety certification**.

## Quick start

```bash
pip install -e .
trustlens demo --out reports      # 3 demo audits -> reports/index.html
pytest -q
```

Audit your own model:

```python
from trustlens import audit_classifier, render_html, model_card_md

res = audit_classifier(
    "RadioScreen v1", model.predict_proba,          # any callable returning probabilities
    X_cal, y_cal, X_test, y_test,                    # calibration split (never used for training) + test split
    X_ref=X_train, feature_names=cols,               # optional: enables drift analysis
    groups_test=site_labels,                         # optional: subgroup parity
    alpha=0.1, target_risk=0.05,
)
open("radioscreen.html", "w").write(render_html(res))
open("MODEL_CARD.md", "w").write(model_card_md(res))
print(res.trust_score, res.verdict, res.oversight)   # e.g. review cases with confidence < 0.63
```

Regression: `audit_regressor(name, model.predict, X_cal, y_cal, X_test, y_test, X_ref=X_train)`.

## Demo results (proxy data)

The three demos use **public or synthetic stand-ins**, not the real project models, to show what the report reveals:

| Demo | Setup | Trust Score | What TrustLens surfaced |
|---|---|---|---|
| RadioScreen (proxy) | Wisconsin diagnostic data, gradient boosting | ~80 | Good selective safety; mild miscalibration and subgroup gap |
| AEGIS-Net (proxy) | 7% attack rate + simulated traffic drift | ~42 | Conformal coverage collapses under shift; major PSI on shifted features |
| Axiom-Zero (proxy) | Time-ordered KPI series with a regime break | ~58 | Coverage drops and drift is flagged after the structural break |

Open `reports/index.html` after running the demo (also published via GitHub Pages workflow).

## Design principles

- **Human oversight is a first-class output**: the escalation threshold is computed, not guessed.
- **Model-agnostic**: only needs `predict_proba` / `predict`; no framework lock-in.
- **Honest by construction**: conformal guarantees require exchangeable calibration/test data; TrustLens reports when this breaks instead of hiding it.
- **Minimal dependencies**: numpy, pandas, scipy, scikit-learn, matplotlib.

## Limitations

- Conformal guarantees are marginal, not per-subgroup or per-instance.
- Permutation importance is a global, correlation-sensitive explanation; it is not causal.
- PSI/KS detect covariate shift only, not label or concept shift.
- The Trust Score weights all pillars equally; adapt weights to your risk context.
- Not a substitute for clinical, security or regulatory validation.

## Project layout

```
trustlens/  calibration.py  uncertainty.py  drift.py  explain.py  report.py  demos.py  cli.py
tests/      unit + end-to-end tests
.github/    CI (3 Python versions) + GitHub Pages demo
```

## References

Guo et al. 2017, *On Calibration of Modern Neural Networks* · Angelopoulos & Bates 2021, *A Gentle Introduction to Conformal Prediction* · Geifman & El-Yaniv 2017, *Selective Classification* · Mitchell et al. 2019, *Model Cards for Model Reporting*.

---

## بالعربية

**TrustLens** طبقة تقييم وثقة مستقلة عن النموذج للذكاء الاصطناعي الخاضع للإشراف البشري. تعطيها دالة `predict` ومجموعة معايرة واختبار، فتُنتج تقريراً يبيّن: هل ثقة النموذج صادقة؟ ما مقدار عدم اليقين؟ **أي الحالات يجب أن يراجعها إنسان؟** هل تغيّرت البيانات؟ وما العوامل التي يعتمد عليها النموذج؟

التشغيل: `pip install -e .` ثم `trustlens demo --out reports` وافتح `reports/index.html`.
> ملاحظة: نتائج العرض التجريبي على بيانات بديلة (عامة أو اصطناعية) لتوضيح طريقة العمل، وليست نتائج نماذج المشاريع الحقيقية. درجة الثقة مقياس إرشادي شفاف وليست شهادة سلامة.
