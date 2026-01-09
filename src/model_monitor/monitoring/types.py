from typing import TypedDict

from model_monitor.core.decisions import DecisionType


class MetricRecord(TypedDict):
    timestamp: float
    batch_id: str
    n_samples: int

    accuracy: float
    f1: float
    avg_confidence: float
    drift_score: float
    decision_latency_ms: float

    action: DecisionType
    reason: str

    previous_model: str | None
    new_model: str | None
