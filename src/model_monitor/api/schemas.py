"""Pydantic request and response models for all API endpoints.

DecisionType is imported from core.decisions (the canonical StrEnum definition)
rather than being redefined here as a Literal.  The StrEnum values are strings
so Pydantic validates incoming JSON action fields against the enum values exactly
as it would against a Literal - ``"retrain"`` is accepted, ``"retrainX"`` is not.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from model_monitor.core.decisions import DecisionType


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
    # New monitoring fields - optional for backward compatibility with older DB rows
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    calibration_error: float | None = None
    output_drift_score: float | None = None
    data_quality_score: float | None = None
    conformal_coverage: float | None = None
    conformal_set_size: float | None = None
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
    """
    Batch result payload for POST /metrics/ingest.

    All metric fields are range-validated at the API boundary.  Invalid
    values are rejected with HTTP 422 before they can reach the trust score
    formula or the decision engine.
    """

    batch_id: str = Field(..., min_length=1, max_length=256)
    n_samples: int = Field(..., ge=1)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    drift_score: float = Field(..., ge=0.0)
    decision_latency_ms: float = Field(..., ge=0.0)
    action: DecisionType
    reason: str = Field(..., min_length=1, max_length=512)
    previous_model: str | None = None
    new_model: str | None = None
