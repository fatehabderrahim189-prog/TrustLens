import tempfile
import unittest
from pathlib import Path

import numpy as np

from trustlens import (apply_temperature, audit_classifier, audit_regressor, conformal_threshold,
                       drift_report, expected_calibration_error, fit_temperature, log_loss, model_card_md,
                       permutation_importance, prediction_sets, psi, render_html, risk_coverage_curve,
                       threshold_for_risk)
from trustlens.uncertainty import set_coverage


class TestCalibration(unittest.TestCase):
    def test_perfect_calibration_has_low_ece(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 20000)
        y = (rng.uniform(size=20000) < p).astype(int)
        self.assertLess(expected_calibration_error(y, p), 0.02)

    def test_overconfident_has_high_ece_and_temperature_fixes_it(self):
        rng = np.random.default_rng(1)
        n = 6000
        logit = rng.normal(0, 1.5, n)
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
        p = 1 / (1 + np.exp(-3 * logit))                  # true logit inflated x3
        before = expected_calibration_error(y, p)
        T = fit_temperature(y, p)
        after = expected_calibration_error(y, apply_temperature(p, T))
        self.assertGreater(before, 0.05)
        self.assertGreater(T, 1.5)
        self.assertLess(after, before / 2)
        self.assertLess(log_loss(y, apply_temperature(p, T)), log_loss(y, p))


class TestUncertainty(unittest.TestCase):
    def test_conformal_coverage_holds_under_exchangeability(self):
        rng = np.random.default_rng(2)
        def draw(n):
            p = rng.dirichlet(np.ones(3), n)
            y = np.array([rng.choice(3, p=r) for r in p])
            return p, y
        pc, yc = draw(1500); pt, yt = draw(3000)
        q = conformal_threshold(pc, yc, alpha=0.1)
        self.assertGreaterEqual(set_coverage(yt, prediction_sets(pt, q)), 0.87)

    def test_risk_coverage_and_threshold(self):
        rng = np.random.default_rng(3)
        conf = rng.uniform(0.5, 1, 3000)
        correct = rng.uniform(size=3000) < conf               # calibrated: higher confidence, fewer errors
        cov, risk = risk_coverage_curve(conf, correct)
        self.assertLess(risk[int(0.3 * len(risk))], risk[-1])
        t = threshold_for_risk(conf, correct, 0.10)
        self.assertLessEqual(t["risk"], 0.10)
        self.assertGreater(t["coverage"], 0.0)

    def test_threshold_for_risk_respects_requested_limit(self):
        rng = np.random.default_rng(15)
        conf = rng.uniform(0.5, 1.0, 3000)
        correct = rng.uniform(size=3000) < conf

        result = threshold_for_risk(conf, correct, 0.20)

        self.assertLessEqual(result["risk"], 0.20)
        self.assertGreater(result["coverage"], 0.0)

    def test_prediction_sets_have_expected_coverage(self):
        rng = np.random.default_rng(9)
        p = rng.dirichlet(np.ones(3), 4000)
        y = np.array([rng.choice(3, p=row) for row in p])
        q = conformal_threshold(p[:2000], y[:2000], alpha=0.2)
        coverage = set_coverage(y[2000:], prediction_sets(p[2000:], q))
        self.assertGreaterEqual(coverage, 0.75)
        self.assertLessEqual(coverage, 0.99)

    def test_prediction_sets_respect_class_count(self):
        rng = np.random.default_rng(10)
        p = rng.dirichlet(np.array([0.2, 1.0, 3.0, 0.5]), 1000)
        y = np.array([rng.choice(4, p=row) for row in p])
        q = conformal_threshold(p[:500], y[:500], alpha=0.1)
        sets = prediction_sets(p[500:], q)

        self.assertEqual(len(sets), 500)
        self.assertTrue(all(1 <= len(s) <= 4 for s in sets))


class TestDriftAndExplain(unittest.TestCase):
    def test_psi_same_vs_shifted(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(size=5000), rng.normal(size=5000)
        self.assertLess(psi(a, b), 0.05)
        self.assertGreater(psi(a, b + 1.0), 0.25)
    def test_psi_identical_distributions_is_near_zero(self):
        rng = np.random.default_rng(11)
        a = rng.normal(size=5000)

        self.assertLess(psi(a, a), 1e-12)

    def test_psi_is_nonnegative(self):
        rng = np.random.default_rng(12)
        a = rng.normal(size=5000)
        b = rng.normal(size=5000)

        self.assertGreaterEqual(psi(a, b), 0.0)
    def test_drift_report_flags_shifted_feature(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(2000, 3))
        Y = rng.normal(size=(2000, 3))
        Y[:, 1] += 1.5

        r = drift_report(X, Y, ["a", "b", "c"])
        self.assertEqual(r.feature[0], "b")
        self.assertEqual(r.status[0], "major")

    def test_drift_report_preserves_feature_names(self):
        rng = np.random.default_rng(13)
        X = rng.normal(size=(1000, 3))
        Y = rng.normal(size=(1000, 3))
        feature_names = ["age", "income", "score"]

        r = drift_report(X, Y, feature_names)

        self.assertEqual(list(r.feature), feature_names)
      
    def test_permutation_importance_finds_informative_feature(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(1500, 4)); y = (X[:, 2] > 0).astype(int)
        imp = permutation_importance(lambda Z: (Z[:, 2] > 0).astype(int), X, y, lambda a, b: float((a == b).mean()),
                                     ["f0", "f1", "f2", "f3"])
        self.assertEqual(imp.feature[0], "f2")
        self.assertGreater(imp.importance[0], 0.3)

    def test_permutation_importance_keeps_irrelevant_feature_low(self):
        rng = np.random.default_rng(14)
        X = rng.normal(size=(1500, 3))
        y = (X[:, 0] > 0).astype(int)

        imp = permutation_importance(
            lambda Z: (Z[:, 0] > 0).astype(int),
            X,
            y,
            lambda a, b: float((a == b).mean()),
            ["signal", "noise1", "noise2"],
        )

        self.assertEqual(imp.feature[0], "signal")
        self.assertLess(imp.importance[1], 0.1)
        self.assertLess(imp.importance[2], 0.1)


class TestEndToEnd(unittest.TestCase):
    def test_classifier_audit_and_render(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(7)
        X = rng.normal(size=(3000, 5)); y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.7, 3000) > 0).astype(int)
        m = LogisticRegression().fit(X[:1000], y[:1000])
        r = audit_classifier("toy", m.predict_proba, X[1000:2000], y[1000:2000], X[2000:], y[2000:], X_ref=X[:1000],
                             groups_test=rng.choice(["a", "b"], 1000))
        self.assertTrue(0 <= r.trust_score <= 100)
        self.assertGreater(r.trust_score, 60)                 # well-specified model on iid data
        html_doc, card = render_html(r), model_card_md(r)
        self.assertIn("Trust", html_doc); self.assertIn("Model Card", card)

    def test_regressor_flags_shift(self):
        from sklearn.linear_model import LinearRegression
        rng = np.random.default_rng(8)
        X = rng.normal(size=(3000, 3)); y = X @ [1.0, 2.0, 0.5] + rng.normal(0, 0.3, 3000)
        m = LinearRegression().fit(X[:1000], y[:1000])
        ok = audit_regressor("iid", m.predict, X[1000:2000], y[1000:2000], X[2000:], y[2000:])
        y_shift = y[2000:] + 3.0
        bad = audit_regressor("shift", m.predict, X[1000:2000], y[1000:2000], X[2000:], y_shift)
        self.assertGreater(ok.pillars["Uncertainty coverage"], 80)
        self.assertLess(bad.pillars["Uncertainty coverage"], ok.pillars["Uncertainty coverage"])

    def test_cli_demo_writes_reports(self):
        from trustlens.cli import run_demo
        with tempfile.TemporaryDirectory() as d:
            run_demo(Path(d))
            for f in ("index.html", "summary.json", "aegis.html", "axiom_model_card.md", "radioscreen.html"):
                self.assertTrue((Path(d) / f).exists(), f)


if __name__ == "__main__":
    unittest.main()
