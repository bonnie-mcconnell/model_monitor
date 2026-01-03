from typing import Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    ready: bool
    reason: Optional[str] = None


class MetricsRecordResponse(BaseModel):
    timestamp: float
    batch_id: str
    n_samples: int
    accuracy: float
    f1: float
    avg_confidence: float
    drift_score: float
    action: str
    reason: str


class MetricsSummaryResponse(BaseModel):
    accuracy: float | None = None
    f1: float | None = None
    avg_confidence: float | None = None
    drift_score: float | None = None


class MetricsEventIn(BaseModel):
    batch_id: str
    n_samples: int
    accuracy: float
    f1: float
    avg_confidence: float
    drift_score: float
    decision_latency_ms: float
    action: str
    reason: str
    previous_model: Optional[str] = None
    new_model: Optional[str] = None
