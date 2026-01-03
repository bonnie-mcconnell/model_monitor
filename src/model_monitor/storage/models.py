from sqlalchemy import Column, Float, Integer, String, Index
from model_monitor.storage.db import Base


class MetricsRecordORM(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True)
    timestamp = Column(Float, index=True)

    batch_id = Column(String, index=True)
    n_samples = Column(Integer)

    accuracy = Column(Float)
    f1 = Column(Float)
    avg_confidence = Column(Float)
    drift_score = Column(Float)
    decision_latency_ms = Column(Float)

    action = Column(String, index=True)
    reason = Column(String)

    previous_model = Column(String, nullable=True)
    new_model = Column(String, nullable=True)


Index("idx_metrics_action_ts", MetricsRecordORM.action, MetricsRecordORM.timestamp)


class MetricsSummaryORM(Base):
    __tablename__ = "metrics_summary"

    id = Column(Integer, primary_key=True)
    window = Column(String, nullable=False)  # e.g. "5m", "1h", "24h"

    n_batches = Column(Integer, nullable=False)

    avg_accuracy = Column(Float)
    avg_f1 = Column(Float)
    avg_confidence = Column(Float)
    avg_drift_score = Column(Float)
    avg_latency_ms = Column(Float)

    last_updated_ts = Column(Float, nullable=False)


Index("ix_metrics_summary_window", MetricsSummaryORM.window)
