from __future__ import annotations

from collections import deque
from typing import Iterator, Optional

from model_monitor.core.decisions import Decision
from model_monitor.storage.decision_store import DecisionStore


class DecisionHistory:
    """
    In-memory decision history buffer.

    Optionally mirrors writes to persistent storage.
    """

    def __init__(
        self,
        maxlen: int = 1000,
        store: Optional[DecisionStore] = None,
    ):
        self._records: deque[Decision] = deque(maxlen=maxlen)
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
        self._records.append(decision)

        if self._store is not None:
            self._store.record(
                decision=decision,
                batch_index=batch_index,
                trust_score=trust_score,
                f1=f1,
                drift_score=drift_score,
                model_version=model_version,
            )

    def __iter__(self) -> Iterator[Decision]:
        return iter(self._records)

    def tail(self, limit: int = 100) -> list[Decision]:
        return list(self._records)[-limit:]


# TODO: ADD/merge files etc

from __future__ import annotations

from collections import deque
from typing import Deque, List

from model_monitor.core.decisions import Decision, DecisionType


class DecisionHistory:
    """
    In-memory rolling decision history.
    """

    def __init__(self, maxlen: int = 50):
        self._decisions: Deque[Decision] = deque(maxlen=maxlen)

    def record(self, decision: Decision) -> None:
        self._decisions.append(decision)

    def recent_actions(self) -> List[DecisionType]:
        return [d.action for d in self._decisions]
