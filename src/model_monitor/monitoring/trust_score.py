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
    """

    acc = clamp(accuracy)
    f1s = clamp(f1)
    conf = clamp(avg_confidence)

    # Drift: higher drift => lower trust
    drift = clamp(1.0 - drift_score)

    # Latency penalty (soft degradation after 500ms)
    latency_penalty = clamp(1.0 - (decision_latency_ms / 1000.0))

    components: TrustScoreComponents = {
        "accuracy": acc,
        "f1": f1s,
        "confidence": conf,
        "drift": drift,
        "latency": latency_penalty,
    }

    weights = {
        "accuracy": 0.30,
        "f1": 0.25,
        "confidence": 0.15,
        "drift": 0.20,
        "latency": 0.10,
    }

    trust = sum(components[k] * weights[k] for k in components)
    trust = clamp(trust)

    return trust, components
