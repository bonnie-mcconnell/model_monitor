from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from model_monitor.core.decisions import Decision
from model_monitor.core.decision_explanation import DecisionExplanation


@dataclass(frozen=True)
class DecisionSnapshot:
    """
    Immutable snapshot of a decision event.

    Used for:
    - audits
    - replay
    - simulation
    """

    batch_index: int
    decision: Decision
    explanation: DecisionExplanation

    metrics: Mapping[str, float]
    model_state: Mapping[str, str]
    timestamp: float
