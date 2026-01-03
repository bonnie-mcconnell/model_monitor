from __future__ import annotations

from sqlalchemy import Column, Integer, Float, String, Index

from model_monitor.storage.db import Base


class MetricsSummaryORM(Base):
    __tablename__ = "metrics_summary"

    id = Column(Integer, primary_key=True)
    window = Column(String, nullable=False, unique=True)  # e.g. "5m", "1h", "24h"

    n_batches = Column(Integer, nullable=False)

    avg_accuracy = Column(Float)
    avg_f1 = Column(Float)
    avg_confidence = Column(Float)
    avg_drift_score = Column(Float)
    avg_latency_ms = Column(Float)

    last_updated_ts = Column(Float, nullable=False)


Index("ix_metrics_summary_window", MetricsSummaryORM.window)
