"""Persistence layer for operational decisions - the audit log."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker

from model_monitor.core.decisions import Decision
from model_monitor.storage.db import Base, SessionLocal
from model_monitor.storage.models.decision_record import DecisionRecordORM


class DecisionStore:
    """
    Persistence layer for operational decisions.

    Append-only audit log. Every decision is persisted with its full
    metadata so the audit trail is complete and queryable after the fact.

    Args:
        db_path: optional path to the SQLite database file.  When supplied,
                 a dedicated engine and session factory are created so the
                 store is fully isolated from the process-level default (used
                 in tests that need a clean database per test).  When absent,
                 the module-level ``SessionLocal`` is used (production path).
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is not None:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(f"sqlite:///{path}", future=True)
            Base.metadata.create_all(engine)
            self._session_factory: sessionmaker[Session] = sessionmaker(
                bind=engine, expire_on_commit=False, class_=Session
            )
        else:
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

    def count(self) -> int:
        """
        Return the total number of decisions recorded.

        Used as a stable, incrementing batch_index for the simulate_decision
        endpoint so cooldown arithmetic matches the live aggregation loop.
        """
        session: Session = self._session_factory()
        try:
            return session.query(DecisionRecordORM).count()
        finally:
            session.close()

    def count_by_action(self) -> dict[str, int]:
        """
        Return cumulative decision counts grouped by action type.

        Uses a single SQL GROUP BY query rather than fetching rows into
        Python - correct at any scale and safe for use as a Prometheus
        counter (monotonically increasing, never decreases).

        Returns a dict mapping action string to total count, e.g.::

            {"none": 1200, "retrain": 4, "promote": 2, "rollback": 1}
        """
        session: Session = self._session_factory()
        try:
            rows = (
                session.query(
                    DecisionRecordORM.action,
                    func.count(DecisionRecordORM.id).label("n"),
                )
                .group_by(DecisionRecordORM.action)
                .all()
            )
            return {action: count for action, count in rows}
        finally:
            session.close()

    def query_range(
        self,
        *,
        from_ts: float | None = None,
        to_ts: float | None = None,
    ) -> list[DecisionRecordORM]:
        """Return all decisions within an optional time range, oldest first.

        Both bounds are inclusive.  Passing neither returns all records.
        Used by the ``model-monitor export`` CLI to write the full audit log.

        Args:
            from_ts: Start Unix timestamp (inclusive).  ``None`` = no lower bound.
            to_ts:   End Unix timestamp (inclusive).  ``None`` = no upper bound.

        Returns:
            List of ORM rows expunged from the session, ordered ascending by
            timestamp so exported files are chronological without extra sorting.
        """
        session: Session = self._session_factory()
        try:
            q = session.query(DecisionRecordORM)
            if from_ts is not None:
                q = q.filter(DecisionRecordORM.timestamp >= from_ts)
            if to_ts is not None:
                q = q.filter(DecisionRecordORM.timestamp <= to_ts)
            rows = q.order_by(DecisionRecordORM.timestamp.asc()).all()
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            session.close()
