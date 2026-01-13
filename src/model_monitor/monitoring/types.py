# src/model_monitor/monitoring/types.py
from __future__ import annotations

from typing import TypedDict
from typing import Literal

from model_monitor.core.decisions import DecisionType


MetricName = Literal[
    "accuracy",
    "f1",
    "avg_confidence",
    "drift_score",
    "decision_latency_ms",
]


class MetricRecord(TypedDict):
    """
    Canonical monitoring record.

    Represents the outcome of a decision batch and any resulting
    model action. This is the single source of truth for:
    - monitoring
    - dashboards
    - audits
    - analytics
    """

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
