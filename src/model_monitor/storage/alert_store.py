"""Persistent store for fired alert records."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from model_monitor.storage.db import Base
from model_monitor.storage.models.alert_record import AlertRecordORM


class AlertStore:
    """
    Append-only persistence for alert events.

    Every time ``check_alerts()`` fires an alert, ``AlertStore.record()``
    should be called so the alert history is queryable via the API.

    This turns a log-only alerting system into one with a queryable audit
    trail: operators can ask "how many critical alerts in the last 24h?" and
    get a structured answer rather than grepping log files.
    """

    def __init__(self, db_path: Path | str = "data/metrics/metrics.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        Base.metadata.create_all(self.engine)

        self.Session = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )

    def record(self, *, window: str, severity: str, trust_score: float) -> None:
        """Persist a fired alert."""
        with self.Session() as session:
            try:
                session.add(
                    AlertRecordORM(
                        timestamp=time.time(),
                        window=window,
                        severity=severity,
                        trust_score=trust_score,
                    )
                )
                session.commit()
            except Exception:
                session.rollback()
                raise

    def tail(
        self,
        *,
        limit: int = 100,
        severity: str | None = None,
        window: str | None = None,
        since_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return recent alerts, newest first.

        Args:
            limit:    maximum records to return.
            severity: filter to "warning" or "critical" only.
            window:   filter to a specific aggregation window.
            since_ts: only return alerts after this Unix timestamp.
        """
        with self.Session() as session:
            q = session.query(AlertRecordORM)

            if severity is not None:
                q = q.filter(AlertRecordORM.severity == severity)
            if window is not None:
                q = q.filter(AlertRecordORM.window == window)
            if since_ts is not None:
                q = q.filter(AlertRecordORM.timestamp >= since_ts)

            rows = q.order_by(AlertRecordORM.timestamp.desc()).limit(limit).all()

        return [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "window": r.window,
                "severity": r.severity,
                "trust_score": r.trust_score,
            }
            for r in rows
        ]

    def count_since(self, since_ts: float, *, severity: str | None = None) -> int:
        """Count alerts since a timestamp, optionally filtered by severity."""
        with self.Session() as session:
            q = session.query(AlertRecordORM).filter(
                AlertRecordORM.timestamp >= since_ts
            )
            if severity is not None:
                q = q.filter(AlertRecordORM.severity == severity)
            return q.count()
