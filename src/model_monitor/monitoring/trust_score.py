from __future__ import annotations
from typing import TypedDict


class TrustScoreComponents(TypedDict):
    accuracy: float
    f1: float
    confidence: float
    drift: float
    latency: float


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


def compute_trust_score(
    *,
    accuracy: float,
    f1: float,
    avg_confidence: float,
    drift_score: float,
    decision_latency_ms: float,
) -> tuple[float, TrustScoreComponents]:
    """
    Compute a bounded, explainable trust score in [0, 1].

    This score is designed for:
    - alerting
    - dashboards
    - retraining policy
    """

    components: TrustScoreComponents = {
        "accuracy": clamp(accuracy),
        "f1": clamp(f1),
        "confidence": clamp(avg_confidence),
        "drift": drift_to_trust(drift_score),
        "latency": latency_score(decision_latency_ms),
    }

    weights = {
        "accuracy": 0.30,
        "f1": 0.25,
        "confidence": 0.15,
        "drift": 0.20,
        "latency": 0.10,
    }

    trust = sum(components[k] * weights[k] for k in components)
    return clamp(trust), components
