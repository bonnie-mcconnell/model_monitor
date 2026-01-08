from __future__ import annotations

from pathlib import Path
from typing import List, Optional, cast

from sqlalchemy import and_, create_engine, or_
from sqlalchemy.orm import sessionmaker

from model_monitor.monitoring.types import DecisionType, MetricRecord
from model_monitor.storage.db import Base
from model_monitor.storage.models.metrics_models import MetricsRecordORM


Cursor = tuple[float, int]  # (timestamp, row id)


class MetricsStore:
    """
    SQLite-backed persistent metrics store.

    Single source of truth for:
    - monitoring
    - analytics
    - dashboards
    - audits
    """

    def __init__(self, db_path: Path | str = "data/metrics/metrics.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            future=True,
        )

        # NOTE: schema creation is intentionally here for now.
        # This will move to startup / migrations later.
        Base.metadata.create_all(self.engine)

        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    # --------------------------------------------------
    # Write
    # --------------------------------------------------
    def write(self, record: MetricRecord) -> None:
        with self.Session() as session:
            session.add(
                MetricsRecordORM(
                    timestamp=record["timestamp"],
                    batch_id=record["batch_id"],
                    n_samples=record["n_samples"],
                    accuracy=record["accuracy"],
                    f1=record["f1"],
                    avg_confidence=record["avg_confidence"],
                    drift_score=record["drift_score"],
                    decision_latency_ms=record["decision_latency_ms"],
                    action=record["action"],
                    reason=record["reason"],
                    previous_model=record["previous_model"],
                    new_model=record["new_model"],
                )
            )
            session.commit()

    # --------------------------------------------------
    # Read (simple)
    # --------------------------------------------------
    def tail(self, *, limit: int = 100) -> List[MetricRecord]:
        """
        Return the most recent metrics in chronological order.
        """
        with self.Session() as session:
            rows: list[MetricsRecordORM] = (
                session.query(MetricsRecordORM)
                .order_by(
                    MetricsRecordORM.timestamp.desc(),
                    MetricsRecordORM.id.desc(),
                )
                .limit(limit)
                .all()
            )

        return [self._to_record(r) for r in reversed(rows)]

    def latest(self) -> Optional[MetricRecord]:
        rows = self.tail(limit=1)
        return rows[0] if rows else None

    # --------------------------------------------------
    # Read (paginated + filtered)
    # --------------------------------------------------
    def list(
        self,
        *,
        limit: int = 50,
        cursor: Optional[Cursor] = None,
        action: Optional[DecisionType] = None,
        model: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        min_f1: Optional[float] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> tuple[List[MetricRecord], Optional[Cursor]]:
        """
        Cursor-based pagination over metrics.

        Returns:
            (records, next_cursor)
        """
        with self.Session() as session:
            q = session.query(MetricsRecordORM)

            # ---- Filters ----
            if action is not None:
                q = q.filter(MetricsRecordORM.action == action)

            if model is not None:
                q = q.filter(
                    or_(
                        MetricsRecordORM.previous_model == model,
                        MetricsRecordORM.new_model == model,
                    )
                )

            if min_accuracy is not None:
                q = q.filter(MetricsRecordORM.accuracy >= min_accuracy)

            if min_f1 is not None:
                q = q.filter(MetricsRecordORM.f1 >= min_f1)

            if start_ts is not None:
                q = q.filter(MetricsRecordORM.timestamp >= start_ts)

            if end_ts is not None:
                q = q.filter(MetricsRecordORM.timestamp <= end_ts)

            # ---- Cursor ----
            if cursor is not None:
                ts, row_id = cursor
                q = q.filter(
                    or_(
                        MetricsRecordORM.timestamp > ts,
                        and_(
                            MetricsRecordORM.timestamp == ts,
                            MetricsRecordORM.id > row_id,
                        ),
                    )
                )

            # ---- Order ----
            q = q.order_by(
                MetricsRecordORM.timestamp.asc(),
                MetricsRecordORM.id.asc(),
            )

            rows: list[MetricsRecordORM] = q.limit(limit + 1).all()

        has_more = len(rows) > limit
        rows = rows[:limit]

        records = [self._to_record(r) for r in rows]

        next_cursor: Optional[Cursor] = None
        if has_more and rows:
            last = rows[-1]
            next_cursor = (
                cast(float, last.timestamp),
                cast(int, last.id),
            )

        return records, next_cursor

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    @staticmethod
    def _to_record(r: MetricsRecordORM) -> MetricRecord:
        return {
            "timestamp": r.timestamp,
            "batch_id": r.batch_id,
            "n_samples": r.n_samples,
            "accuracy": r.accuracy,
            "f1": r.f1,
            "avg_confidence": r.avg_confidence,
            "drift_score": r.drift_score,
            "decision_latency_ms": r.decision_latency_ms,
            "action": cast(DecisionType, r.action),
            "reason": r.reason,
            "previous_model": r.previous_model,
            "new_model": r.new_model,
        }
