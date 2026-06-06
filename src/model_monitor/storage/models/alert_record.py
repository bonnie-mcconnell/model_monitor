"""ORM model for the alert history log."""

from __future__ import annotations

from sqlalchemy import Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from model_monitor.storage.db import Base


class AlertRecordORM(Base):
    """
    Append-only log of every alert that fired.

    Stores enough context to answer: "how many critical alerts in the
    last 24h?" and "what was the trust score when it fired?".
    """

    __tablename__ = "alert_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False, index=True)

    # Aggregation window that triggered the alert (e.g. "5m", "1h", "24h")
    window: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # "warning" or "critical"
    severity: Mapped[str] = mapped_column(String, nullable=False, index=True)

    trust_score: Mapped[float] = mapped_column(Float, nullable=False)


Index("ix_alert_history_ts_severity", AlertRecordORM.timestamp, AlertRecordORM.severity)
