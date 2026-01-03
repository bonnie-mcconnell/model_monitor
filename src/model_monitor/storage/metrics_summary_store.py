from __future__ import annotations

import time
from sqlalchemy.orm import Session

from model_monitor.storage.db import SessionLocal
from model_monitor.storage.metrics_summary import MetricsSummaryORM


class MetricsSummaryStore:
    """
    Persistence layer for aggregated metric summaries.
    One row per rolling window (e.g. 5m, 1h, 24h).
    """

    def __init__(self):
        self._session_factory = SessionLocal

    def upsert(
        self,
        window: str,
        *,
        n_batches: int,
        avg_accuracy: float | None,
        avg_f1: float | None,
        avg_confidence: float | None,
        avg_drift_score: float | None,
        avg_latency_ms: float | None,
    ) -> None:
        session: Session = self._session_factory()
        try:
            row = (
                session.query(MetricsSummaryORM)
                .filter(MetricsSummaryORM.window == window)
                .one_or_none()
            )

            if row is None:
                row = MetricsSummaryORM(window=window)
                session.add(row)

            row.n_batches = n_batches
            row.avg_accuracy = avg_accuracy
            row.avg_f1 = avg_f1
            row.avg_confidence = avg_confidence
            row.avg_drift_score = avg_drift_score
            row.avg_latency_ms = avg_latency_ms
            row.last_updated_ts = float(time.time())  # type: ignore[assignment]

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
