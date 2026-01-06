from __future__ import annotations

from typing import Any, Sequence
from collections import Counter

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
        actions = [str(r["action"]) for r in self.history.read_all()]
        return dict(Counter(actions))

    def decision_tail(self, limit: int = 100) -> Sequence[dict[str, Any]]:
        records = self.history.read_all()
        return records[-limit:]
