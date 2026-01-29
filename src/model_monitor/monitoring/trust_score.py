from __future__ import annotations

from typing import TypedDict, Literal, Tuple


class TrustScoreComponents(TypedDict):
    accuracy: float
    f1: float
    confidence: float
    drift: float
    latency: float


TrustComponentKey = Literal[
    "accuracy",
    "f1",
    "confidence",
    "drift",
    "latency",
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


def compute_trust_score(
    *,
    accuracy: float,
    f1: float,
    avg_confidence: float,
    drift_score: float,
    decision_latency_ms: float,
) -> Tuple[float, TrustScoreComponents]:
    """
    Compute a bounded, explainable trust score in [0, 1].

    Designed for:
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

    weights: dict[TrustComponentKey, float] = {
        "accuracy": 0.30,
        "f1": 0.25,
        "confidence": 0.15,
        "drift": 0.20,
        "latency": 0.10,
    }

    trust: float = sum(
        components[key] * weights[key]
        for key in weights
    )

    return clamp(trust), components


# TODO: add behavioural components e.g
"""
behavioral_penalty = clamp(
    1.0 - behavioral_violation_rate * cfg.behavior.weight,
    min=0.0,
    max=1.0,
)

"""