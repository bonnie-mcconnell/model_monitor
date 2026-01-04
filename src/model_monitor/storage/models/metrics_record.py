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
