"""ORM model for current rolling metric summaries (one row per window)."""
from __future__ import annotations

from sqlalchemy import Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from model_monitor.storage.db import Base


class MetricsSummaryORM(Base):
    __tablename__ = "metrics_summary"

    id: Mapped[int] = mapped_column(primary_key=True)

    window: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    n_batches: Mapped[int] = mapped_column(Integer, nullable=False)

    avg_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    avg_f1: Mapped[float] = mapped_column(Float, nullable=False)
    avg_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    avg_drift_score: Mapped[float] = mapped_column(Float, nullable=False)
    avg_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    trust_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    updated_ts: Mapped[float] = mapped_column(Float, nullable=False)


Index("ix_metrics_summary_window", MetricsSummaryORM.window)