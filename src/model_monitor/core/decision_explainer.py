"""Human-readable explanation layer for decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decisions import Decision, DecisionType


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

        rule_map: dict[DecisionType, str] = {
            DecisionType.REJECT: "severe_drift",
            DecisionType.ROLLBACK: "catastrophic_regression",
            DecisionType.RETRAIN: "sustained_degradation",
            DecisionType.PROMOTE: "stability_promotion",
            DecisionType.SYSTEM_ERROR: "circuit_breaker_open",
        }

        rule = rule_map.get(action, "within_thresholds")

        return ExplainedDecision(
            summary=decision.reason,
            rule_triggered=rule,
            contributing_factors=dict(decision.metadata),
        )
