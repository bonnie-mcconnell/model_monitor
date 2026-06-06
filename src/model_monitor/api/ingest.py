"""
Ingest endpoint for external inference pipelines.

POST /metrics/ingest accepts a batch result and writes a MetricRecord to
the store. This is what makes the system connectable to a real inference
pipeline rather than just the internal simulation.

Authentication: API key checked against the MONITOR_API_KEY environment
variable. A missing or incorrect key returns 401. No key configured means
the endpoint is disabled entirely - this prevents accidental open ingestion
in environments where the variable has not been set.
"""

from __future__ import annotations

import os
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from model_monitor.api.schemas import MetricsEventIn
from model_monitor.core.decisions import DecisionType
from model_monitor.monitoring.types import MetricRecord
from model_monitor.storage.metrics_store import MetricsStore

router = APIRouter(prefix="/metrics", tags=["ingest"])

_ENV_KEY = "MONITOR_API_KEY"

# Lazy singleton: created on first request, not at import time.
# Mirrors the pattern in storage/model_store.py - prevents the import
# side-effect of creating the SQLite file and running Base.metadata.create_all()
# in environments that only import this module for type-checking or routing.
_metrics_store: MetricsStore | None = None


def _get_metrics_store() -> MetricsStore:
    global _metrics_store
    if _metrics_store is None:
        _metrics_store = MetricsStore()
    return _metrics_store


def _require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """
    FastAPI dependency that enforces API key authentication on the ingest endpoint.

    Reads the expected key from the MONITOR_API_KEY environment variable at
    request time (not at startup) so the variable can be rotated without
    restarting the process.

    The distinction between 503 and 401 is intentional:

    - **503 Service Unavailable**: the ``MONITOR_API_KEY`` environment variable
      is not set at all.  The endpoint is administratively disabled - no key
      will ever be accepted.  A 503 correctly signals "this service is not
      available" rather than "your credentials are wrong."  Operators who have
      not configured the key get a clear message rather than a misleading 401.

    - **401 Unauthorized**: the key is configured but the provided value does
      not match.  Standard auth-failure semantics per RFC 7235.

    Raises:
        HTTPException 503: if ``MONITOR_API_KEY`` is not set.
        HTTPException 401: if the provided key does not match the configured value.
    """
    expected = os.environ.get(_ENV_KEY)

    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Ingest endpoint is disabled: {_ENV_KEY} environment "
                "variable is not set."
            ),
        )

    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


@router.post(
    "/ingest",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_require_api_key)],
    summary="Ingest a batch metric record from an external inference pipeline",
)
def ingest_metrics(payload: MetricsEventIn) -> dict[str, object]:
    """
    Write one batch result to the metrics store.

    The caller is expected to have already computed all metric values.
    This endpoint does not perform inference, drift computation, or
    decision evaluation - it records what the caller reports.

    The aggregation loop running in the background will pick up the
    persisted record on its next pass and fold it into rolling summaries.
    """
    record: MetricRecord = {
        "timestamp": time.time(),
        "batch_id": payload.batch_id,
        "n_samples": payload.n_samples,
        "accuracy": payload.accuracy,
        "f1": payload.f1,
        "avg_confidence": payload.avg_confidence,
        "drift_score": payload.drift_score,
        "decision_latency_ms": payload.decision_latency_ms,
        # External pipelines report an aggregate drift score but not per-feature
        # breakdowns or calibration - those require data held server-side.
        "calibration_error": None,
        "feature_drift_scores": None,
        # External pipelines do not run behavioral contracts server-side.
        "behavioral_violation_rate": None,
        # SHAP attribution requires the model and reference data server-side.
        "causal_drift_report": None,
        "mmd_p_value": None,
        "mmd_is_drift": None,
        "shap_attribution": None,
        # New monitoring fields - not available via external ingest.
        "p95_latency_ms": None,
        "p99_latency_ms": None,
        "output_drift_score": None,
        "output_drift_class_scores": None,
        "data_quality_score": None,
        "conformal_coverage": None,
        "conformal_set_size": None,
        "action": DecisionType(payload.action),
        "reason": payload.reason,
        "previous_model": payload.previous_model,
        "new_model": payload.new_model,
    }

    _get_metrics_store().write(record)

    return {
        "accepted": True,
        "batch_id": payload.batch_id,
        "timestamp": record["timestamp"],
    }
