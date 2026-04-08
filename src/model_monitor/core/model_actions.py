"""ModelAction enum — the executable form of a Decision."""
from __future__ import annotations

from enum import Enum

from model_monitor.core.decisions import DecisionType


class ModelAction(str, Enum):
    """
    Executable model lifecycle actions.
    """

    NONE = "none"
    RETRAIN = "retrain"
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    REJECT = "reject"

    @classmethod
    def from_decision(cls, decision_action: DecisionType) -> ModelAction:
        """
        Explicit boundary between policy decisions and executable actions.
        """
        try:
            return cls(decision_action)
        except ValueError as exc:
            raise ValueError(
                f"Decision action '{decision_action}' cannot be executed"
            ) from exc
