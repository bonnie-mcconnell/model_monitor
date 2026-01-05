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
        records = self.history.read_all()
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
        records = self.history.read_all()
        if not records:
            return []

        df = pd.DataFrame(records).tail(limit)

        # Ensure JSON-safe + string keys
        rows: list[dict[str, Any]] = []
        for record in df.to_dict(orient="records"):
            rows.append({str(k): v for k, v in record.items()})

        return rows
