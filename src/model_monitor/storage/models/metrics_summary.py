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

    # Average ECE across batches in this window.  Null when no calibrated
    # batches are present (e.g. no ground-truth labels in the window).
    avg_calibration_error: Mapped[float | None] = mapped_column(Float, nullable=True)

    # New monitoring signal aggregates - all nullable when the corresponding
    # monitor is not configured or no data has been collected yet.
    avg_output_drift_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_data_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_conformal_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_conformal_set_size: Mapped[float | None] = mapped_column(Float, nullable=True)

    updated_ts: Mapped[float] = mapped_column(Float, nullable=False)


Index("ix_metrics_summary_window", MetricsSummaryORM.window)
