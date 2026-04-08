"""ORM model for the decision snapshot write-ahead log."""
from __future__ import annotations

from sqlalchemy import Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from model_monitor.storage.db import Base


class SnapshotRecordORM(Base):
    """
    Write-ahead log for DecisionSnapshot execution state.

    Written BEFORE execution begins. On restart, the executor reads
    incomplete snapshots (status='pending') and skips retrains whose
    retrain_key is already present, preventing duplicate retrains after
    a crash.

    The retrain_key is a SHA-256 fingerprint of the evidence DataFrame.
    It is content-addressed and collision-resistant: two identical
    evidence windows produce the same key, two different windows do not.
    """

    __tablename__ = "decision_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)

    decision_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    action: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False, index=True)

    # Set when a retrain is triggered, before execution begins.
    # NULL for non-retrain actions.
    retrain_key: Mapped[str | None] = mapped_column(String, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)


Index("ix_snapshot_retrain_key", SnapshotRecordORM.retrain_key)
