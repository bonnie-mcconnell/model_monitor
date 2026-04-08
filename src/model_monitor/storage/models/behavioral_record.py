"""ORM model for behavioral contract evaluation outcomes."""
from __future__ import annotations

from sqlalchemy import Float, String
from sqlalchemy.orm import Mapped, mapped_column

from model_monitor.storage.db import Base


class BehavioralRecordORM(Base):
    """
    Persisted summary of a single behavioral contract evaluation.

    Append-only. Never updated or deleted.

    We store the minimal set of fields needed to compute violation_rate:
    outcome and created_at. model_id and decision_id are stored for
    provenance and deduplication - not for querying by violation logic.
    """

    __tablename__ = "behavioral_evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)

    # UUID from BehavioralContractRunner - enforced unique for idempotency
    decision_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    model_id: Mapped[str] = mapped_column(String, nullable=False)

    # "accept" | "warn" | "block" - stored as string to survive schema evolution
    outcome: Mapped[str] = mapped_column(String, nullable=False)

    # Unix epoch float - indexed for time-window queries
    created_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)
