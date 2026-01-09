from __future__ import annotations

from collections import deque
from typing import Iterator

from model_monitor.core.decisions import Decision


class DecisionHistory:
    """
    In-memory decision history buffer.

    Purpose:
    - Capture recent operational decisions
    - Support dashboards, APIs, and analytics
    - Explicit write path, read-only iteration

    This is intentionally ephemeral.
    Persistence can be layered on later if needed.
    """

    def __init__(self, maxlen: int = 1000):
        self._records: deque[Decision] = deque(maxlen=maxlen)

    def record(self, decision: Decision) -> None:
        """Record a new decision."""
        self._records.append(decision)

    def __iter__(self) -> Iterator[Decision]:
        return iter(self._records)

    def tail(self, limit: int = 100) -> list[Decision]:
        """Return the most recent decisions."""
        return list(self._records)[-limit:]
