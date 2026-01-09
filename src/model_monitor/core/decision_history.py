from __future__ import annotations

from collections import deque
from typing import Iterator

from model_monitor.core.decision_record import DecisionRecord

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
        self._records: deque[DecisionRecord] = deque(maxlen=maxlen)

    def record(self, record: DecisionRecord) -> None:
        """Record a new decision."""
        self._records.append(record)

    def __iter__(self) -> Iterator[DecisionRecord]:
        return iter(self._records)

    def tail(self, limit: int = 100) -> list[DecisionRecord]:
        """Return the most recent decisions."""
        return list(self._records)[-limit:]
