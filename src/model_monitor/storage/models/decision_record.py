from __future__ import annotations

from sqlalchemy import String, Float, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column

from model_monitor.storage.db import Base


class DecisionRecordORM(Base):
    """
    Persistent audit log of operational decisions.

    Append-only.
    Never updated or deleted.
    """

    __tablename__ = "decision_history"

    id: Mapped[int] = mapped_column(primary_key=True)

    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    batch_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    action: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False)

    trust_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    drift_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    model_version: Mapped[str | None] = mapped_column(String, nullable=True)


Index(
    "ix_decision_history_ts",
    DecisionRecordORM.timestamp,
)
