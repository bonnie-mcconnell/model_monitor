from __future__ import annotations
from typing import Any, Sequence
import pandas as pd

from model_monitor.core.decision_history import DecisionHistory


class DecisionAnalytics:
    """
    Read-only analytics over decision history.

    Used by:
    - dashboards
    - APIs
    - offline analysis
    """

    def __init__(self, history: DecisionHistory) -> None:
        self.history = history

    def decision_summary(self) -> dict[str, int]:
        """
        Return a summary count of actions.
        """
        records = list(self.history)
        if not records:
            return {}

        df = pd.DataFrame(records)

        if "action" not in df.columns:
            return {}

        # Ensure keys are str to satisfy mypy
        raw_summary = df["action"].astype(str).value_counts().to_dict()
        summary: dict[str, int] = {str(k): int(v) for k, v in raw_summary.items()}
        return summary

    def decision_tail(self, limit: int = 100) -> Sequence[dict[str, Any]]:
        """
        Return the last N decision records as JSON-safe dicts.
        """
        records = list(self.history)
        if not records:
            return []

        df = pd.DataFrame(records).tail(limit)

        # Explicitly return JSON-safe dicts
        return [{str(k): v for k, v in row.items()} for row in df.to_dict(orient="records")]
