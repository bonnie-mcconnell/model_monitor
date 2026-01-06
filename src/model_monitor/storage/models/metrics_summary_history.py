# TODO: check if needed
from __future__ import annotations

from sqlalchemy import Float, Integer, String, Index
from sqlalchemy.orm import Mapped, mapped_column

from model_monitor.storage.db import Base


class MetricsSummaryHistoryORM(Base):
    __tablename__ = "metrics_summary_history"

    id: Mapped[int] = mapped_column(primary_key=True)

    window: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)

    n_batches: Mapped[int] = mapped_column(Integer, nullable=False)

    avg_accuracy: Mapped[float] = mapped_column(Float)
    avg_f1: Mapped[float] = mapped_column(Float)
    avg_confidence: Mapped[float] = mapped_column(Float)
    avg_drift_score: Mapped[float] = mapped_column(Float)
    avg_latency_ms: Mapped[float] = mapped_column(Float)


Index(
    "ix_metrics_summary_history_window_ts",
    MetricsSummaryHistoryORM.window,
    MetricsSummaryHistoryORM.timestamp,
)
