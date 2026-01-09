from __future__ import annotations

import time
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from model_monitor.storage.db import SessionLocal
from model_monitor.storage.models.decision_record import DecisionRecordORM
from model_monitor.core.decisions import Decision


class DecisionStore:
    """
    Persistence layer for operational decisions.

    Append-only audit log.
    """

    def __init__(self) -> None:
        self._session_factory = SessionLocal

    def record(
        self,
        *,
        decision: Decision,
        batch_index: int | None = None,
        trust_score: float | None = None,
        f1: float | None = None,
        drift_score: float | None = None,
        model_version: str | None = None,
    ) -> None:
        session: Session = self._session_factory()
        try:
            row = DecisionRecordORM(
                timestamp=time.time(),
                batch_index=batch_index,
                action=decision.action,
                reason=decision.reason,
                trust_score=trust_score,
                f1=f1,
                drift_score=drift_score,
                model_version=model_version,
            )
            session.add(row)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def tail(self, limit: int = 100) -> list[DecisionRecordORM]:
        session: Session = self._session_factory()
        try:
            return (
                session.query(DecisionRecordORM)
                .order_by(DecisionRecordORM.timestamp.desc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()
