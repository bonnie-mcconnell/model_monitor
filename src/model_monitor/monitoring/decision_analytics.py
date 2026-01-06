from typing import Any, Sequence
from model_monitor.monitoring.decision_history import DecisionHistory, DecisionRecord


class DecisionAnalytics:
    """
    Read-only analytics over decision history. 
    Used by dashboard, API, offline access.
    """

    def __init__(self, history: DecisionHistory):
        self.history = history

    def decision_summary(self) -> dict[str, int]:
        actions = [str(r["action"]) for r in self.history]
        counts: dict[str, int] = {}
        for action in actions:
            counts[action] = counts.get(action, 0) + 1
        return counts

    def decision_tail(self, limit: int = 100) -> Sequence[DecisionRecord]:
        records = list(self.history)
        return records[-limit:]
