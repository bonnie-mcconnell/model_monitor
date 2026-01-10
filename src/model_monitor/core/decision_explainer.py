from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

from model_monitor.core.decisions import Decision
from model_monitor.core.decision_snapshot import DecisionSnapshot


@dataclass(frozen=True)
class ExplainedDecision:
    """
    Human-readable, structured explanation for a decision.

    Intended for:
    - dashboards
    - audits
    - user-facing APIs
    """

    summary: str
    rule_triggered: str
    contributing_factors: Mapping[str, Any]


class DecisionExplainer:
    """
    Derives explanations from decisions + snapshots.

    This layer is:
    - non-authoritative
    - presentation-focused
    - side-effect free
    """

    def explain(
        self,
        *,
        decision: Decision,
        snapshot: DecisionSnapshot,
    ) -> ExplainedDecision:
        action = decision.action

        if action == "reject":
            rule = "severe_drift"
        elif action == "rollback":
            rule = "catastrophic_regression"
        elif action == "retrain":
            rule = "sustained_degradation"
        elif action == "promote":
            rule = "stability_promotion"
        else:
            rule = "within_thresholds"

        return ExplainedDecision(
            summary=decision.reason,
            rule_triggered=rule,
            contributing_factors={
                "trust_score": snapshot.trust_score,
                "f1": snapshot.f1,
                "f1_baseline": snapshot.f1_baseline,
                "drift_score": snapshot.drift_score,
                "batch_index": snapshot.batch_index,
            },
        )
