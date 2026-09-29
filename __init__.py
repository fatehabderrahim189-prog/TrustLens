"""TrustLens: a model-agnostic trust & evaluation layer for human-governed AI."""
from .calibration import (apply_temperature, brier_score, expected_calibration_error,
                          fit_temperature, log_loss, reliability_curve)
from .drift import drift_report, psi
from .explain import permutation_importance, subgroup_report
from .report import AuditResult, audit_classifier, audit_regressor, model_card_md, render_html
from .uncertainty import (bootstrap_ci, conformal_interval_width, conformal_threshold,
                          prediction_sets, risk_coverage_curve, threshold_for_risk)

__version__ = "0.1.0"
__all__ = [n for n in dir() if not n.startswith("_")]
