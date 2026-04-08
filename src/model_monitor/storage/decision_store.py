"""Persistence layer for operational decisions - the audit log."""
from __future__ import annotations

import json
import time

from sqlalchemy.orm import Session

from model_monitor.core.decisions import Decision
from model_monitor.storage.db import SessionLocal
from model_monitor.storage.models.decision_record import DecisionRecordORM


class DecisionStore:
    """
    Persistence layer for operational decisions.

    Append-only audit log. Every decision is persisted with its full
    metadata so the audit trail is complete and queryable after the fact.
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
        """
        Persist a decision to the audit log.

        Decision.metadata is serialised as JSON so the full context
        (baseline_f1, thresholds, cooldown state) is recoverable from
        the audit trail without reconstructing it from other tables.
        """
        metadata_json: str | None = None
        if decision.metadata:
            try:
                metadata_json = json.dumps(decision.metadata)
            except (TypeError, ValueError):
                # Non-serialisable metadata is a caller bug; log it as a
                # string rather than silently dropping the decision record.
                metadata_json = json.dumps({"_raw": str(decision.metadata)})

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
                metadata_json=metadata_json,
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
            rows = (
                session.query(DecisionRecordORM)
                .order_by(DecisionRecordORM.timestamp.desc())
                .limit(limit)
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            session.close()
