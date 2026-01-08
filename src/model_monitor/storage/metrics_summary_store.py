from __future__ import annotations

import time
from typing import Optional

from sqlalchemy.orm import Session

from model_monitor.storage.db import SessionLocal
from model_monitor.storage.models.metrics_summary import MetricsSummaryORM


class MetricsSummaryStore:
    """
    Persistence layer for rolling aggregated metric summaries.

    One row per aggregation window (e.g. "5m", "1h", "24h").
    Overwritten on each aggregation pass.

    Note:
        Schema creation is handled centrally (db.py / startup),
        not in this store.
    """

    def __init__(self) -> None:
        self._session_factory = SessionLocal

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
        now = time.time()

        try:
            row = (
                session.query(MetricsSummaryORM)
                .filter(MetricsSummaryORM.window == window)
                .one_or_none()
            )

            if row is None:
                row = MetricsSummaryORM(
                    window=window,
                    n_batches=n_batches,
                    updated_ts=now,
                )
                session.add(row)

            row.n_batches = n_batches
            row.avg_accuracy = avg_accuracy
            row.avg_f1 = avg_f1
            row.avg_confidence = avg_confidence
            row.avg_drift_score = avg_drift_score
            row.avg_latency_ms = avg_latency_ms
            row.updated_ts = now

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get(self, window: str) -> Optional[MetricsSummaryORM]:
        session: Session = self._session_factory()
        try:
            return (
                session.query(MetricsSummaryORM)
                .filter(MetricsSummaryORM.window == window)
                .one_or_none()
            )
        finally:
            session.close()
