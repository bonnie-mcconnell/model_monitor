from __future__ import annotations

from sqlalchemy.orm import Session

from model_monitor.storage.db import SessionLocal
from model_monitor.storage.models.metrics_summary_history import (
    MetricsSummaryHistoryORM,
)


class MetricsSummaryHistoryStore:
    """
    Append-only persistence layer for historical aggregated metric summaries.

    One row per aggregation window per aggregation run.
    Used for dashboards, trend analysis, and audits.
    """

    def __init__(self) -> None:
        self._session_factory = SessionLocal

    def write(
        self,
        *,
        window: str,
        timestamp: float,
        n_batches: int,
        avg_accuracy: float,
        avg_f1: float,
        avg_confidence: float,
        avg_drift_score: float,
        avg_latency_ms: float,
    ) -> None:
        """
        Persist a historical snapshot of aggregated metrics.

        This method is intentionally append-only.
        """
        session: Session = self._session_factory()
        try:
            session.add(
                MetricsSummaryHistoryORM(
                    window=window,
                    timestamp=timestamp,
                    n_batches=n_batches,
                    avg_accuracy=avg_accuracy,
                    avg_f1=avg_f1,
                    avg_confidence=avg_confidence,
                    avg_drift_score=avg_drift_score,
                    avg_latency_ms=avg_latency_ms,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
