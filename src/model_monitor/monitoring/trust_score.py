"""Bounded trust score computed from performance and behavioral signals."""
from __future__ import annotations

from typing import Literal, TypedDict


class TrustScoreComponents(TypedDict):
    accuracy: float
    f1: float
    confidence: float
    drift: float
    latency: float
    behavioral: float


TrustComponentKey = Literal[
    "accuracy",
    "f1",
    "confidence",
    "drift",
    "latency",
    "behavioral",
]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def latency_score(ms: float) -> float:
    """
    Soft latency degradation:
    - <=300ms: no penalty
    - 300–1500ms: linear decay
    - >=1500ms: floored
    """
    if ms <= 300:
        return 1.0
    if ms >= 1500:
        return 0.0
    return clamp(1.0 - (ms - 300) / 1200.0)


def drift_to_trust(drift_score: float) -> float:
    """
    Convert drift signal into trust.
    Assumes drift_score is PSI-like where:
    - <0.1 = negligible
    - 0.1–0.2 = moderate
    - >0.2 = severe
    """
    if drift_score <= 0.1:
        return 1.0
    if drift_score >= 0.3:
        return 0.0
    return clamp(1.0 - (drift_score - 0.1) / 0.2)


def behavioral_score(violation_rate: float) -> float:
    """
    Convert behavioral violation rate into a trust component in [0, 1].

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
    behavioral_violation_rate: float = 0.0,
    behavioral_weight: float = 0.15,
) -> tuple[float, TrustScoreComponents]:
    """
    Compute a bounded, explainable trust score in [0, 1].

    The score is a weighted sum of five performance components plus a
    behavioral component derived from contract evaluation results.

    The behavioral component is additive rather than a post-hoc penalty so
    that its contribution is transparent in TrustScoreComponents and auditable
    in dashboards alongside the other components. The existing performance
    weights are scaled down proportionally to accommodate it, preserving their
    relative importance while keeping the total weight at 1.0.

    Args:
        behavioral_violation_rate: proportion of recent contract evaluations
            that resulted in BLOCK or WARN. Defaults to 0.0 so callers that
            have not yet integrated behavioral monitoring are unaffected.
        behavioral_weight: weight assigned to the behavioral component.
            Defaults to 0.15. Set to 0.0 to disable behavioural influence.
    """
    behavioral_weight = clamp(behavioral_weight, lo=0.0, hi=1.0)
    remaining = 1.0 - behavioral_weight

    # Performance weights, scaled so they sum to `remaining`.
    # Relative proportions are preserved: accuracy is still the most
    # important performance signal, latency is still the least.
    _base_weights: dict[str, float] = {
        "accuracy":   0.30,
        "f1":         0.25,
        "confidence": 0.15,
        "drift":      0.20,
        "latency":    0.10,
    }
    base_total = sum(_base_weights.values())  # 1.0 - but explicit for safety

    components: TrustScoreComponents = {
        "accuracy":   clamp(accuracy),
        "f1":         clamp(f1),
        "confidence": clamp(avg_confidence),
        "drift":      drift_to_trust(drift_score),
        "latency":    latency_score(decision_latency_ms),
        "behavioral": behavioral_score(behavioral_violation_rate),
    }

    # Explicit sum so mypy can verify every key is a valid TypedDict literal.
    # The weights are scaled to sum to (1 - behavioral_weight),
    # preserving the relative importance of each performance component.
    scale = remaining / base_total
    trust: float = (
        components["accuracy"]   * _base_weights["accuracy"]   * scale
        + components["f1"]         * _base_weights["f1"]         * scale
        + components["confidence"] * _base_weights["confidence"] * scale
        + components["drift"]      * _base_weights["drift"]      * scale
        + components["latency"]    * _base_weights["latency"]    * scale
        + components["behavioral"] * behavioral_weight
    )

    return clamp(trust), components
