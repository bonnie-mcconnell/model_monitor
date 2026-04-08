"""Read-only analytics over decision history for dashboards and APIs."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from model_monitor.core.decision_history import DecisionHistory


class DecisionAnalytics:
    """
    Read-only analytics over decision history.

    Uses only the standard library - no pandas - so this layer stays
    lightweight and importable without the 50MB pandas dependency in
    contexts that only need counts and tail access.

    Used by:
    - dashboards
    - APIs
    - offline analysis
    """

    def __init__(self, history: DecisionHistory) -> None:
        self.history = history

    def decision_summary(self) -> dict[str, int]:
        """
        Return a count of each action type seen in history.

        Returns an empty dict when history is empty.
        """
        records = list(self.history)
        if not records:
            return {}
        return dict(Counter(r.action for r in records))

    def decision_tail(self, limit: int = 100) -> Sequence[dict[str, Any]]:
        """
        Return the last ``limit`` decision records as JSON-safe dicts.

        Records are returned in chronological order (oldest first within
        the window) matching the convention used by MetricsStore.tail().
        """
        records = list(self.history)
        if not records:
            return []
        tail = records[-limit:] if limit < len(records) else records
        return [
            {
                "action": r.action,
                "reason": r.reason,
                **{str(k): v for k, v in r.metadata.items()},
            }
            for r in tail
        ]
