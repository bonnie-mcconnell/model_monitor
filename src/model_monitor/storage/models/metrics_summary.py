from __future__ import annotations

from typing import Optional
from sqlalchemy import Float, Integer, String, Index
from sqlalchemy.orm import Mapped, mapped_column

from model_monitor.storage.db import Base


class MetricsSummaryORM(Base):
    __tablename__ = "metrics_summary"

    id: Mapped[int] = mapped_column(primary_key=True)

    window: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    n_batches: Mapped[int] = mapped_column(Integer, nullable=False)

    avg_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    avg_f1: Mapped[Optional[float]] = mapped_column(Float)
    avg_confidence: Mapped[Optional[float]] = mapped_column(Float)
    avg_drift_score: Mapped[Optional[float]] = mapped_column(Float)
    avg_latency_ms: Mapped[Optional[float]] = mapped_column(Float)

    updated_ts: Mapped[float] = mapped_column(Float, nullable=False)


Index("ix_metrics_summary_window", MetricsSummaryORM.window)
