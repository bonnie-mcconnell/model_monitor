"""Append-only history of aggregated metric summaries for trend analysis."""

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
        from collections.abc import Callable

        self._session_factory: Callable[[], Session] = SessionLocal

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

    def list_history(
        self,
        *,
        window: str,
        limit: int = 100,
    ) -> list[MetricsSummaryHistoryORM]:
        """Return up to ``limit`` history rows for a given aggregation window,
        ordered oldest-first so callers receive a ready-to-plot time series.

        Rows are expunged from the session before return so callers can access
        column values after this method returns without holding an open session.
        """
        session: Session = self._session_factory()
        try:
            rows = (
                session.query(MetricsSummaryHistoryORM)
                .filter(MetricsSummaryHistoryORM.window == window)
                .order_by(MetricsSummaryHistoryORM.timestamp.desc())
                .limit(limit)
                .all()
            )
            for row in rows:
                session.expunge(row)
            return list(reversed(rows))
        finally:
            session.close()

    def query_range(
        self,
        *,
        window: str,
        from_ts: float,
        to_ts: float,
    ) -> list[MetricsSummaryHistoryORM]:
        """Return history rows for a given window within a Unix-timestamp range.

        Results are ordered oldest-first so they can be replayed chronologically
        by ``cli/replay.py`` without additional sorting.  Both bounds are
        inclusive.

        Args:
            window:  Aggregation window label (``"5m"``, ``"1h"``, ``"24h"``).
            from_ts: Start of the range (Unix timestamp, inclusive).
            to_ts:   End of the range (Unix timestamp, inclusive).

        Returns:
            List of ORM rows expunged from the session.  Empty when no records
            fall within the requested range.
        """
        session: Session = self._session_factory()
        try:
            rows = (
                session.query(MetricsSummaryHistoryORM)
                .filter(
                    MetricsSummaryHistoryORM.window == window,
                    MetricsSummaryHistoryORM.timestamp >= from_ts,
                    MetricsSummaryHistoryORM.timestamp <= to_ts,
                )
                .order_by(MetricsSummaryHistoryORM.timestamp.asc())
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            session.close()
