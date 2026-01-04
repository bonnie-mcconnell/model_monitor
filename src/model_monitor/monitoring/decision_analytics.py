from __future__ import annotations

from typing import Any, Mapping, Sequence
import pandas as pd

from model_monitor.monitoring.decision_history import DecisionHistory


class DecisionAnalytics:
    """
    Read-only analytics over decision history.

    Used by:
    - dashboards
    - APIs
    - offline analysis
    """

    def __init__(self, history: DecisionHistory):
        self.history = history

    def decision_summary(self) -> Mapping[str, int]:
        records = self.history.read_all()
        if not records:
            return {}

        df = pd.DataFrame(records)
        return df["action"].value_counts().to_dict()

    def decision_tail(self, limit: int = 100) -> Sequence[Mapping[str, Any]]:
        records = self.history.read_all()
        if not records:
            return []

        df = pd.DataFrame(records).tail(limit)
        return df.to_dict(orient="records")
