from __future__ import annotations
from model_monitor.core.decisions import Decision
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decision_explanation import DecisionExplanation


class DecisionExplainer:
    """
    Derives explanations from decisions + snapshots.
    """

    def explain(
        self,
        *,
        decision: Decision,
        snapshot: DecisionSnapshot,
    ) -> DecisionExplanation:
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

        return DecisionExplanation(
            summary=decision.reason,
            rule_triggered=rule,
            contributing_factors={
                "trust_score": snapshot.trust_score,
                "f1": snapshot.f1,
                "f1_baseline": snapshot.f1_baseline,
                "drift_score": snapshot.drift_score,
            },
        )
