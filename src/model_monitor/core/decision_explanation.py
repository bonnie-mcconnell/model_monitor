from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from model_monitor.core.decisions import Decision, DecisionType


@dataclass(frozen=True)
class DecisionExplanation:
    """
    Human- and machine-readable explanation of a system decision.

    Designed for:
    - dashboards
    - audits
    - offline replay
    - incident reports
    """

    action: DecisionType
    reason: str

    signals: Mapping[str, float]
    thresholds: Mapping[str, float]
    context: Mapping[str, str]

    @classmethod
    def from_decision(
        cls,
        *,
        decision: Decision,
        signals: Mapping[str, float],
        thresholds: Mapping[str, float],
        context: Mapping[str, str] | None = None,
    ) -> "DecisionExplanation":
        return cls(
            action=decision.action,
            reason=decision.reason,
            signals=signals,
            thresholds=thresholds,
            context=context or {},
        )
