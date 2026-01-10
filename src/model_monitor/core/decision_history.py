from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, Optional

from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.storage.decision_store import DecisionStore


class DecisionHistory:
    """
    In-memory rolling decision history.

    - Maintains a bounded in-memory buffer for fast access
    - Optionally mirrors decisions to persistent storage
    - Serves as the single source of truth for recent actions
    """

    def __init__(
        self,
        maxlen: int = 100,
        store: Optional[DecisionStore] = None,
    ) -> None:
        self._decisions: Deque[Decision] = deque(maxlen=maxlen)
        self._store = store

    def record(
        self,
        decision: Decision,
        *,
        batch_index: int | None = None,
        trust_score: float | None = None,
        f1: float | None = None,
        drift_score: float | None = None,
        model_version: str | None = None,
    ) -> None:
        """
        Record a decision in memory and (optionally) persistent storage.
        """
        self._decisions.append(decision)

        if self._store is not None:
            self._store.record(
                decision=decision,
                batch_index=batch_index,
                trust_score=trust_score,
                f1=f1,
                drift_score=drift_score,
                model_version=model_version,
            )

    def recent_actions(self, limit: Optional[int] = None) -> list[DecisionType]:
        """
        Return recent decision actions (most recent last).

        Used by decision engines to reason about action history
        and prevent oscillations.
        """
        decisions = (
            list(self._decisions)
            if limit is None
            else list(self._decisions)[-limit:]
        )
        return [d.action for d in decisions]

    def tail(self, limit: int = 100) -> list[Decision]:
        """
        Return the last `limit` decisions.
        """
        return list(self._decisions)[-limit:]

    def __iter__(self) -> Iterator[Decision]:
        return iter(self._decisions)
