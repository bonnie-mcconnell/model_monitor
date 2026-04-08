"""Write-ahead log for DecisionSnapshot - enables crash-safe retrain deduplication."""
from __future__ import annotations

from sqlalchemy.orm import Session

from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.storage.db import SessionLocal
from model_monitor.storage.models.snapshot_record import SnapshotRecordORM


class SnapshotStore:
    """
    Write-ahead persistence for DecisionSnapshot execution state.

    The critical invariant: a snapshot is written to the database
    BEFORE its corresponding action is executed. On restart, incomplete
    snapshots can be detected and their retrain_keys used to suppress
    duplicate retrains.

    Without this store, a crash between `snapshot.retrain_key = key`
    and `snapshot.status = 'executed'` loses the key and allows a
    duplicate retrain on the next startup.

    Usage pattern in the executor:
        1. snapshot_store.write(snapshot)          # before execution
        2. execute the action
        3. snapshot_store.update_status(snapshot)  # after completion
    """

    def __init__(self) -> None:
        self._session_factory = SessionLocal

    def write(self, snapshot: DecisionSnapshot) -> None:
        """
        Persist a snapshot before execution begins.

        Idempotent: a second write for the same decision_id is silently
        ignored. The first write wins - this matches the executor's
        retrain_key idempotency semantics.
        """
        session: Session = self._session_factory()
        try:
            exists = (
                session.query(SnapshotRecordORM)
                .filter(SnapshotRecordORM.decision_id == snapshot.decision_id)
                .first()
            )
            if exists is not None:
                return

            session.add(SnapshotRecordORM(
                decision_id=snapshot.decision_id,
                action=snapshot.action,
                status=snapshot.status,
                timestamp=snapshot.timestamp,
                retrain_key=snapshot.retrain_key,
                model_version=snapshot.model_version,
            ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def update_status(self, snapshot: DecisionSnapshot) -> None:
        """
        Update the persisted status after execution completes or fails.

        Raises if the snapshot was never written (programming error:
        update_status must only be called after write).
        """
        session: Session = self._session_factory()
        try:
            row = (
                session.query(SnapshotRecordORM)
                .filter(SnapshotRecordORM.decision_id == snapshot.decision_id)
                .first()
            )
            if row is None:
                raise RuntimeError(
                    f"Cannot update status for unknown snapshot {snapshot.decision_id}. "
                    "write() must be called before update_status()."
                )
            row.status = snapshot.status
            row.retrain_key = snapshot.retrain_key
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def is_retrain_key_known(self, retrain_key: str) -> bool:
        """
        Return True if this retrain_key was already used in a prior execution.

        Called on startup to detect crashed retrains. If True, the executor
        must skip the retrain - the evidence window was already consumed and
        the retrain either completed or failed before the crash.
        """
        session: Session = self._session_factory()
        try:
            return (
                session.query(SnapshotRecordORM)
                .filter(SnapshotRecordORM.retrain_key == retrain_key)
                .first()
            ) is not None
        finally:
            session.close()

    def pending_retrains(self) -> list[SnapshotRecordORM]:
        """
        Return snapshots that were written but never completed.

        Called on startup for crash recovery: pending snapshots indicate
        a retrain that started but did not finish. Their retrain_keys
        can be used to skip duplicate execution.
        """
        session: Session = self._session_factory()
        try:
            return (
                session.query(SnapshotRecordORM)
                .filter(
                    SnapshotRecordORM.status == "pending",
                    SnapshotRecordORM.action == "retrain",
                )
                .all()
            )
        finally:
            session.close()
