"""ORM model registry: import all models so Base.metadata discovers every table."""
from __future__ import annotations

# Import all ORM models so Base.metadata.create_all() discovers every table.
from model_monitor.storage.models.behavioral_record import BehavioralRecordORM
from model_monitor.storage.models.decision_record import DecisionRecordORM
from model_monitor.storage.models.metrics_models import MetricsRecordORM
from model_monitor.storage.models.metrics_summary import MetricsSummaryORM
from model_monitor.storage.models.metrics_summary_history import (
    MetricsSummaryHistoryORM,
)
from model_monitor.storage.models.snapshot_record import SnapshotRecordORM

__all__ = [
    "MetricsRecordORM",
    "MetricsSummaryORM",
    "MetricsSummaryHistoryORM",
    "DecisionRecordORM",
    "BehavioralRecordORM",
    "SnapshotRecordORM",
]
