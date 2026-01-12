from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

from model_monitor.core.decisions import Decision
from model_monitor.core.decision_snapshot import DecisionSnapshot


@dataclass(frozen=True)
class ExplainedDecision:
    """
    Human-readable explanation for a decision.
    """

    summary: str
    rule_triggered: str
    contributing_factors: Mapping[str, Any]


class DecisionExplainer:
    """
    Presentation-only explanation layer.
    """

    def explain(
        self,
        *,
        decision: Decision,
        snapshot: DecisionSnapshot,
    ) -> ExplainedDecision:
        action = decision.action

        rule_map = {
            "reject": "severe_drift",
            "rollback": "catastrophic_regression",
            "retrain": "sustained_degradation",
            "promote": "stability_promotion",
        }

        rule = rule_map.get(action, "within_thresholds")

        return ExplainedDecision(
            summary=decision.reason,
            rule_triggered=rule,
            contributing_factors=dict(decision.metadata),
        )
