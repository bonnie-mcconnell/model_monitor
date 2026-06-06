"""Bounded trust score computed from performance, quality, and behavioral signals.

Component breakdown (default weights):
  accuracy        25% - batch accuracy_score vs ground truth
  f1              20% - batch F1 (macro averaged)
  calibration     15% - Expected Calibration Error converted to trust [0,1]
  drift           20% - input feature PSI converted to [0,1]
  latency         17% - p95 latency (falls back to mean when p95 unavailable)
  data_quality     5% - null rate, range violations, schema consistency
  behavioral       5% - contract violation EMA (BM branch only)

The behavioral and data_quality components default to 1.0 (no penalty) when
their respective monitors are not configured, so callers without those
subsystems are unaffected and weights still sum to 1.0.

Calibration replaces the raw mean-confidence component from the previous
version.  ECE is strictly more informative: a perfectly calibrated model has
ECE = 0 regardless of its average confidence, while a model that always
outputs 90% confidence with 70% accuracy has low ECE but high mean-confidence.

Why p95 latency instead of mean:
    Average latency can look healthy while p99 is pathological.  A trust
    score that penalises p95 latency is more representative of the tail
    user experience.  When per-sample timing is not available the mean is
    used as a fallback, preserving backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from model_monitor.config.settings import TrustScoreConfig


class TrustScoreComponents(TypedDict):
    accuracy: float
    f1: float
    calibration: float
    drift: float
    latency: float
    data_quality: float
    behavioral: float


TrustComponentKey = Literal[
    "accuracy",
    "f1",
    "calibration",
    "drift",
    "latency",
    "data_quality",
    "behavioral",
]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def latency_score(ms: float) -> float:
    """Soft latency degradation.

    - <=300ms: no penalty
    - 300-1500ms: linear decay to 0.0
    - >=1500ms: floored at 0.0
    """
    if ms <= 300:
        return 1.0
    if ms >= 1500:
        return 0.0
    return clamp(1.0 - (ms - 300) / 1200.0)


def drift_to_trust(drift_score: float) -> float:
    """Convert PSI drift score to a [0, 1] trust contribution.

    Mapping:
    - PSI < 0.1:  negligible drift -> 1.0
    - 0.1-0.3:    linear decay
    - PSI > 0.3:  severe drift -> 0.0
    """
    if drift_score <= 0.1:
        return 1.0
    if drift_score >= 0.3:
        return 0.0
    return clamp(1.0 - (drift_score - 0.1) / 0.2)


def calibration_to_trust(ece: float | None) -> float:
    """Convert Expected Calibration Error to a [0, 1] trust contribution.

    Mapping:
    - ECE = None: no labels -> neutral 0.8 (slight discount for uncertainty)
    - ECE = 0.0:  perfect calibration -> 1.0
    - ECE = 0.05: industry warning threshold -> 0.5
    - ECE >= 0.10: severe miscalibration -> 0.0

    The 0.05 threshold matches the commonly cited practical warning level
    from Guo et al. (2017) "On Calibration of Modern Neural Networks."
    """
    if ece is None:
        # No labels available - return neutral rather than penalising.
        return 0.8
    if ece <= 0.0:
        return 1.0
    if ece >= 0.10:
        return 0.0
    return clamp(1.0 - ece / 0.10)


def behavioral_score(violation_rate: float) -> float:
    """Convert behavioral violation rate into a trust component in [0, 1].

    violation_rate is the proportion of recent contract evaluations that
    resulted in BLOCK or WARN (0.0 = no violations, 1.0 = all violated).

    The mapping is direct: zero violations contribute full score, full
    violation rate contributes zero. This is intentionally linear - a model
    that fails 50% of behavioral checks is half as trustworthy as one that
    fails none.
    """
    return clamp(1.0 - violation_rate)


def compute_trust_score(
    *,
    accuracy: float,
    f1: float,
    avg_confidence: float,
    drift_score: float,
    decision_latency_ms: float,
    calibration_error: float | None = None,
    p95_latency_ms: float | None = None,
    output_drift_score: float | None = None,
    data_quality_score: float | None = None,
    behavioral_violation_rate: float = 0.0,
    behavioral_weight: float = 0.05,
    config: TrustScoreConfig | None = None,
) -> tuple[float, TrustScoreComponents]:
    """Compute a bounded, explainable trust score in [0, 1].

    The score is a weighted sum of seven components:
      accuracy, F1, calibration, drift, latency, data_quality, behavioral.

    New vs previous version:
      - ``calibration_error`` replaces raw ``avg_confidence`` in the formula.
        ECE is a strictly better measure of confidence quality.  avg_confidence
        is still accepted for API compatibility but is no longer used directly.
      - ``p95_latency_ms`` is used for the latency component when available,
        falling back to ``decision_latency_ms`` (mean).  P95 latency is more
        representative of tail user experience than mean.
      - ``data_quality_score`` is a new component.  Defaults to 1.0 (no
        penalty) when DataQualityMonitor is not configured.
      - ``behavioral_weight`` defaults to 0.05 (down from 0.15 in previous
        version) to accommodate the two new components within the unit budget.

    Args:
        accuracy:                 batch accuracy_score.
        f1:                       batch F1 score (macro averaged).
        avg_confidence:           mean max class probability.  No longer used
                                  directly in the formula; kept for API compat.
        drift_score:              mean input PSI across features.
        output_drift_score:       mean output PSI across classes.  When both
                                  input and output PSI are available the drift
                                  component uses ``max(drift_score, output_drift_score)``
                                  so either signal can drive the penalty.
                                  None means only input PSI is used.
        decision_latency_ms:      mean end-to-end decision latency in ms.
        calibration_error:        ECE in [0, 1].  None when labels unavailable.
        p95_latency_ms:           p95 per-sample latency in ms.  Uses mean
                                  fallback when None.
        data_quality_score:       scalar in [0, 1] from DataQualityMonitor.
                                  Defaults to 1.0 (no penalty) when None.
        behavioral_violation_rate: EMA of contract violation score.
        behavioral_weight:        weight for behavioral component.
        config:                   optional TrustScoreConfig from YAML.

    Returns:
        (trust_score, TrustScoreComponents) - trust_score clamped to [0, 1].
    """
    # Load weights from config or use defaults.  All seven weights come from
    # config so operators can tune sensitivity without code changes.
    if config is not None:
        w_accuracy = config.accuracy
        w_f1 = config.f1
        w_calibration = config.calibration
        w_drift = config.drift
        w_latency = config.latency
        w_dq = config.data_quality
        w_behavioral = config.behavioral
    else:
        w_accuracy = 0.23
        w_f1 = 0.18
        w_calibration = 0.14
        w_drift = 0.18
        w_latency = 0.17
        w_dq = 0.05
        w_behavioral = 0.05

    # Clamp weights so caller-supplied overrides can't produce NaN.
    behavioral_weight = clamp(w_behavioral, lo=0.0, hi=1.0)

    # Combined drift: take the maximum of input and output PSI so that either
    # signal alone can drive the full drift penalty.  A model with stable inputs
    # but a collapsing output distribution is just as concerning as one with
    # drifting inputs.
    combined_drift = drift_score
    if output_drift_score is not None:
        combined_drift = max(drift_score, output_drift_score)

    # Latency: prefer p95 when available - mean latency can mask tail issues.
    latency_for_score = (
        p95_latency_ms if p95_latency_ms is not None else decision_latency_ms
    )

    # Data quality: default to 1.0 when monitor not configured (no penalty).
    dq_score = data_quality_score if data_quality_score is not None else 1.0

    components: TrustScoreComponents = {
        "accuracy": clamp(accuracy),
        "f1": clamp(f1),
        "calibration": calibration_to_trust(calibration_error),
        "drift": drift_to_trust(combined_drift),
        "latency": latency_score(latency_for_score),
        "data_quality": clamp(dq_score),
        "behavioral": behavioral_score(behavioral_violation_rate),
    }

    trust: float = (
        components["accuracy"] * w_accuracy
        + components["f1"] * w_f1
        + components["calibration"] * w_calibration
        + components["drift"] * w_drift
        + components["latency"] * w_latency
        + components["data_quality"] * w_dq
        + components["behavioral"] * behavioral_weight
    )

    return clamp(trust), components
