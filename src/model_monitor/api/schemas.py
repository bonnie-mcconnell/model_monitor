from typing import Optional, List
from pydantic import BaseModel


# --------------------------------------------------
# Health
# --------------------------------------------------

class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    ready: bool
    reason: Optional[str] = None


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
    action: str
    reason: str
    previous_model: Optional[str]
    new_model: Optional[str]


class MetricsSummaryResponse(BaseModel):
    window: str
    timestamp: float
    n_batches: int

    avg_accuracy: Optional[float]
    avg_f1: Optional[float]
    avg_confidence: Optional[float]
    avg_drift_score: Optional[float]
    avg_latency_ms: Optional[float]


class MetricsSummarySeriesResponse(BaseModel):
    window: str
    items: List[MetricsSummaryResponse]


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
    action: str
    reason: str
    previous_model: Optional[str] = None
    new_model: Optional[str] = None
