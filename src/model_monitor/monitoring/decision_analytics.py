from __future__ import annotations

from typing import Any, Sequence
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

    def decision_summary(self) -> dict[str, int]:
        records = list(self.history)
        if not records:
            return {}

        df = pd.DataFrame(records)

        if "action" not in df.columns:
            return {}

        return (
            df["action"]
            .astype(str)
            .value_counts()
            .to_dict()
        )

    def decision_tail(self, limit: int = 100) -> Sequence[dict[str, Any]]:
        records = list(self.history)
        if not records:
            return []

        df = pd.DataFrame(records).tail(limit)

        # Explicitly return JSON-safe dicts
        return [
            {str(k): v for k, v in row.items()}
            for row in df.to_dict(orient="records")
        ]
