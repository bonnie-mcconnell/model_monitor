from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Any


@dataclass(frozen=True)
class DecisionExplanation:
    """
    Structured explanation for a decision.
    """

    summary: str
    contributing_factors: Mapping[str, Any]
    rule_triggered: str
