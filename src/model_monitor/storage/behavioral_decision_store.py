"""Append-only persistence for behavioral contract evaluation outcomes."""
from __future__ import annotations

from sqlalchemy.orm import Session

from model_monitor.contracts.behavioral.records import DecisionRecord
from model_monitor.contracts.outcome import DecisionOutcome
from model_monitor.storage.db import SessionLocal
from model_monitor.storage.models.behavioral_record import BehavioralRecordORM

# Outcomes that constitute a behavioral violation
_VIOLATION_OUTCOMES = {DecisionOutcome.BLOCK, DecisionOutcome.WARN}


class BehavioralDecisionStore:
    """
    Append-only persistence layer for behavioral contract evaluations.

    Stores one row per BehavioralContractRunner evaluation. Provides
    violation_rate() so the aggregation loop can feed behavioral signal
    into the trust score without coupling the contracts system to the
    monitoring system at call time.

    The violation rate is intentionally computed over a time window that
    matches the monitoring aggregation window (5m, 1h, 24h) - same window,
    same denominator, consistent signal.
    """

    def __init__(self) -> None:
        self._session_factory = SessionLocal

    def record(self, decision: DecisionRecord) -> None:
        """
        Persist a behavioral evaluation outcome.

        Silently skips duplicate decision_ids - if the caller retries after
        a network failure the record is not written twice.
        """
        session: Session = self._session_factory()
        try:
            exists = (
                session.query(BehavioralRecordORM)
                .filter(BehavioralRecordORM.decision_id == decision.decision_id)
                .first()
            )
            if exists is not None:
                return

            session.add(
                BehavioralRecordORM(
                    decision_id=decision.decision_id,
                    model_id=decision.context.model_id,
                    outcome=decision.outcome.value,
                    created_at=decision.created_at.timestamp(),
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def violation_rate(self, *, since_ts: float) -> float:
        """
        Return the proportion of evaluations since `since_ts` that resulted
        in BLOCK or WARN.

        Returns 0.0 when no evaluations exist in the window - no data means
        no signal, not maximum suspicion.
        """
        session: Session = self._session_factory()
        try:
            rows = (
                session.query(BehavioralRecordORM)
                .filter(BehavioralRecordORM.created_at >= since_ts)
                .all()
            )
        finally:
            session.close()

        if not rows:
            return 0.0

        violation_values = {o.value for o in _VIOLATION_OUTCOMES}
        violations = sum(1 for r in rows if r.outcome in violation_values)
        return violations / len(rows)
