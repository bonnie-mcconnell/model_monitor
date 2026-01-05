from __future__ import annotations

from typing import Optional
from sqlalchemy import Float, Integer, String, Index
from sqlalchemy.orm import Mapped, mapped_column

from model_monitor.storage.db import Base


class MetricsRecordORM(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[float] = mapped_column(Float, index=True)

    batch_id: Mapped[str] = mapped_column(String, index=True)
    n_samples: Mapped[int] = mapped_column(Integer)

    accuracy: Mapped[float] = mapped_column(Float)
    f1: Mapped[float] = mapped_column(Float)
    avg_confidence: Mapped[float] = mapped_column(Float)
    drift_score: Mapped[float] = mapped_column(Float)
    decision_latency_ms: Mapped[float] = mapped_column(Float)

    action: Mapped[str] = mapped_column(String, index=True)
    reason: Mapped[str] = mapped_column(String)

    previous_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    new_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)


Index("idx_metrics_action_ts", MetricsRecordORM.action, MetricsRecordORM.timestamp)
