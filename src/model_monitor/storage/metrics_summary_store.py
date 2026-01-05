from __future__ import annotations

import time
from typing import Optional

from sqlalchemy.orm import Session

from model_monitor.storage.db import SessionLocal
from model_monitor.storage.models.metrics_summary import MetricsSummaryORM


class MetricsSummaryStore:
    """
    Persistence layer for rolling aggregated metric summaries.

    Design:
    - One row per aggregation window (e.g. "5m", "1h", "24h")
    - Overwritten on each aggregation pass
    - Used for dashboards & real-time monitoring
    """

    def __init__(self):
        self._session_factory = SessionLocal

    # ------------------------------------------------------------------
    # WRITE PATH (used by aggregation.py)
    # ------------------------------------------------------------------

    def upsert(
        self,
        *,
        window: str,
        n_batches: int,
        avg_accuracy: float,
        avg_f1: float,
        avg_confidence: float,
        avg_drift_score: float,
        avg_latency_ms: float,
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
            row.last_updated_ts = time.time()

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # READ PATH (used by dashboard.py)
    # ------------------------------------------------------------------

    def get(self, window: str) -> Optional[MetricsSummaryORM]:
        """
        Fetch the current rolling summary for a given window.
        """
        session: Session = self._session_factory()
        try:
            return (
                session.query(MetricsSummaryORM)
                .filter(MetricsSummaryORM.window == window)
                .one_or_none()
            )
        finally:
            session.close()
