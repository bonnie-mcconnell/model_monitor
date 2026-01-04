from __future__ import annotations

from model_monitor.core.decisions import Decision
from model_monitor.monitoring.decision_history import DecisionHistory


class DecisionLogger:
    """
    Centralized logger for operational decisions.

    Responsibilities:
    - persist decisions
    - enforce consistent logging
    - decouple decision policy from storage
    """

    def __init__(self, history: DecisionHistory):
        self.history = history

    def log(
        self,
        *,
        decision: Decision,
        batch_index: int,
        f1: float,
        f1_baseline: float,
        drift_score: float,
        model_version: str | None,
    ) -> None:
        self.history.write(
            batch_index=batch_index,
            action=decision.action,
            reason=decision.reason,
            f1=f1,
            f1_baseline=f1_baseline,
            drift_score=drift_score,
            model_version=model_version,
        )
