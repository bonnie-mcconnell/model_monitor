"""ORM model for historical aggregated metric snapshots."""

from __future__ import annotations

from sqlalchemy import Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from model_monitor.storage.db import Base


class MetricsSummaryHistoryORM(Base):
    """
    Append-only historical record of aggregated metric summaries.

    One row per aggregation window per aggregation run.
    New monitoring signal columns are nullable so existing databases upgrade
    cleanly - see storage/migrations.py migrations 8 and 9.
    """

    __tablename__ = "metrics_summary_history"

    id: Mapped[int] = mapped_column(primary_key=True)

    window: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)

    n_batches: Mapped[int] = mapped_column(Integer, nullable=False)

    avg_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    avg_f1: Mapped[float] = mapped_column(Float, nullable=False)
    avg_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    avg_drift_score: Mapped[float] = mapped_column(Float, nullable=False)
    avg_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    # New monitoring signal columns - all nullable for backward compatibility.
    avg_output_drift_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_data_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_conformal_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_conformal_set_size: Mapped[float | None] = mapped_column(Float, nullable=True)


Index(
    "ix_metrics_summary_history_window_ts",
    MetricsSummaryHistoryORM.window,
    MetricsSummaryHistoryORM.timestamp,
)
