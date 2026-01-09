from __future__ import annotations
from dataclasses import dataclass
from model_monitor.core.decisions import Decision
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decision_explanation import DecisionExplanation


@dataclass(frozen=True)
class DecisionRecord:
    """
    Immutable audit record.
    """

    decision: Decision
    snapshot: DecisionSnapshot
    explanation: DecisionExplanation
