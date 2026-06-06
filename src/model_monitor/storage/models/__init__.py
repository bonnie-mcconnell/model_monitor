"""ORM model registry: import all models so Base.metadata discovers every table.

Importing this package ensures that every ORM class is registered with
``Base.metadata`` before ``create_all()`` is called.  Without these imports,
tables whose modules have not been imported are silently absent from the schema.

``tests/conftest.py`` imports this package via::

    import model_monitor.storage.models  # noqa: F401

which triggers these imports transitively.  Keeping the imports here (rather
than in conftest) means any code path that creates the schema - migration
scripts, the FastAPI lifespan, manual setup - gets the full table list without
needing to know about conftest.
"""

from __future__ import annotations

from model_monitor.storage.models.decision_record import DecisionRecordORM
from model_monitor.storage.models.metrics_models import MetricsRecordORM
from model_monitor.storage.models.metrics_summary import MetricsSummaryORM
from model_monitor.storage.models.metrics_summary_history import (
    MetricsSummaryHistoryORM,
)

__all__ = [
    "MetricsRecordORM",
    "MetricsSummaryORM",
    "MetricsSummaryHistoryORM",
    "DecisionRecordORM",
]
