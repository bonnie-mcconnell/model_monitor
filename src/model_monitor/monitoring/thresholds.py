"""
Threshold constants used by the in-process alerting layer.

**Important - two distinct constant types live here:**

ALERTING CONSTANTS (``MIN_TRUST_SCORE``, ``CRITICAL_TRUST_SCORE``)
    Imported by ``alerting.py`` and used by ``check_alerts()`` to decide
    whether to emit log alerts.  These are *not* used by the decision engine.
    Change them to tune alert sensitivity without affecting retrain/rollback
    policy.

REFERENCE CONSTANTS (everything else)
    Documentation-only defaults that reflect the values in the YAML config
    files (``drift.yaml``, ``retrain.yaml``, ``rollback.yaml``).  They are
    *not* imported anywhere in the production pipeline - the live system
    reads policy thresholds exclusively from those YAML files via
    ``load_config()``.  They exist here so the intent behind each threshold
    is visible in one place with explanatory comments.

    If you change a YAML value, update the corresponding constant here too
    so they stay in sync as documentation.
"""

from __future__ import annotations

# ======================
# Alerting (imported by alerting.py)
# ======================

# Minimum conformal coverage rate before a WARNING alert fires.
# When empirical coverage drops below (1 - alpha), the model is provably
# making worse-than-expected predictions on the current distribution.
MIN_CONFORMAL_COVERAGE: float = 0.85

# Minimum data quality score before a WARNING alert fires.
# Below this threshold the incoming data has enough issues (nulls, range
# violations, schema mismatches) that downstream metrics should not be trusted.
MIN_DATA_QUALITY_SCORE: float = 0.80

# Maximum output drift PSI before a WARNING alert fires.
# Mirrors the input PSI warning threshold for consistency.
MAX_OUTPUT_DRIFT_SCORE: float = 0.10

# Trust score at which a WARNING-level log alert fires.
# Tune this to control how sensitively the alerting layer reacts to
# gradual degradation.  The decision engine has its own min_f1_gain
# threshold for retraining - these two are independent.
MIN_TRUST_SCORE: float = 0.70

# Trust score at which a CRITICAL-level log alert fires.
# Below this value the model is considered operationally unreliable.
# Corresponds to the threshold below which on-call escalation is appropriate.
CRITICAL_TRUST_SCORE: float = 0.60


# ======================
# Reference (documentation only - not imported by production code)
# ======================

# Mirrors drift.yaml: psi_threshold.
# PSI below 0.1 is negligible; 0.1–0.2 is moderate; above 0.2 triggers reject.
PSI_DRIFT_THRESHOLD: float = 0.2
DRIFT_WINDOW: int = 500  # samples in the DriftMonitor rolling buffer

# Mirrors retrain.yaml: min_f1_gain.
# F1 must drop by at least this much below the promotion-time baseline
# before the engine fires a retrain.
MIN_F1_IMPROVEMENT: float = 0.02
MIN_RETRAIN_SAMPLES: int = 1000
RETRAIN_COOLDOWN_BATCHES: int = 5

# Mirrors rollback.yaml: max_f1_drop.
# A drop larger than this triggers an immediate rollback to the previous model.
MAX_F1_DROP_FOR_ROLLBACK: float = 0.15

# Informational floors - not enforced by the decision engine directly;
# the trust score formula already penalises low accuracy and F1.
MIN_F1: float = 0.75
MIN_ACCURACY: float = 0.80
