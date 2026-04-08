"""Pydantic request and response models for all API endpoints."""
from typing import Any, Literal

from pydantic import BaseModel, Field

DecisionType = Literal[
    "none",
    "retrain",
    "promote",
    "rollback",
    "reject",
    "system_error",
]


class DecisionSchema(BaseModel):
    action: DecisionType
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------
# Health
# --------------------------------------------------

class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    ready: bool
    reason: str | None = None


# --------------------------------------------------
# Metrics
# --------------------------------------------------

class MetricsRecordResponse(BaseModel):
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


class MetricsSummaryResponse(BaseModel):
    window: str
    timestamp: float
    n_batches: int

    avg_accuracy: float | None
    avg_f1: float | None
    avg_confidence: float | None
    avg_drift_score: float | None
    avg_latency_ms: float | None


class MetricsSummarySeriesResponse(BaseModel):
    window: str
    items: list[MetricsSummaryResponse]


# --------------------------------------------------
# Ingestion
# --------------------------------------------------

class MetricsEventIn(BaseModel):
    batch_id: str
    n_samples: int
    accuracy: float
    f1: float
    avg_confidence: float
    drift_score: float
    decision_latency_ms: float
    action: DecisionType
    reason: str
    previous_model: str | None = None
    new_model: str | None = None
