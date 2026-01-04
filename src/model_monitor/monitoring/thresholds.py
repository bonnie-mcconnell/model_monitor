"""
Operational defaults for monitoring & retraining.

These are policy-level values intended for:
- local runs
- experiments
- documentation

Production systems should override via config.
"""

# ======================
# Drift
# ======================
PSI_DRIFT_THRESHOLD: float = 0.2
DRIFT_WINDOW: int = 200


# ======================
# Performance floors
# ======================
MIN_F1: float = 0.75
MIN_ACCURACY: float = 0.80


# ======================
# Trust policy
# ======================
MIN_TRUST_SCORE: float = 0.70
CRITICAL_TRUST_SCORE: float = 0.60


# ======================
# Retraining policy
# ======================
MIN_F1_IMPROVEMENT: float = 0.02
MIN_RETRAIN_SAMPLES: int = 500
RETRAIN_COOLDOWN_BATCHES: int = 5
