# keep 
# ensure never mutates state/only used by api/ui
from __future__ import annotations

from typing import Any, Mapping, Sequence
import pandas as pd

from model_monitor.storage.metrics_store import MetricsStore


class DecisionAnalytics:
    """
    Read-only analytics over decision history.

    Intended for dashboards, APIs, and offline inspection.
    """

    def __init__(self, store: MetricsStore):
        self.store = store

    def decision_summary(self) -> Mapping[str, int]:
        """
        Count decisions by action type.
        """
        records = self.store.tail(limit=10_000)
        if not records:
            return {}

        df = pd.DataFrame(records)
        if "action" not in df.columns:
            return {}

        return df["action"].value_counts().to_dict()

    def decision_tail(self, limit: int = 100) -> Sequence[Mapping[str, Any]]:
        """
        Return recent decision records for UI display.
        """
        records = self.store.tail(limit=limit)
        if not records:
            return []

        df = pd.DataFrame(records)

        normalized: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            normalized.append(
                {str(k): v for k, v in row.to_dict().items()}
            )

        return normalized
