from __future__ import annotations

from model_monitor.core.decisions import Decision
from model_monitor.core.decision_explanation import DecisionExplanation


def explain_decision(
    *,
    decision: Decision,
    trust_score: float,
    f1: float,
    f1_baseline: float,
    drift_score: float,
) -> DecisionExplanation:
    """
    Construct a structured explanation for a decision.

    This is intentionally separate from policy logic.
    """

    return DecisionExplanation.from_decision(
        decision=decision,
        signals={
            "trust_score": trust_score,
            "f1": f1,
            "baseline_f1": f1_baseline,
            "drift_score": drift_score,
        },
        thresholds={
            "psi_threshold": decision.metadata.get("threshold", float("nan")),
        },
        context={
            "engine": "decision_engine_v1",
        },
    )
