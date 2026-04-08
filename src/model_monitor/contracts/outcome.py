"""Decision outcomes and reason codes from behavioral evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DecisionOutcome(Enum):
    """
    Outcome of evaluating a model output against a behavioral contract.

    ACCEPT: all guarantees passed - output is within contract.
    WARN:   one or more HIGH severity failures - flagged but not blocked.
    BLOCK:  at least one CRITICAL failure - output must not be served.

    Note: rollback is an operational action handled by DecisionEngine, not
    by the behavioral contracts pipeline. It is deliberately absent here.
    """
    ACCEPT = "accept"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class OutcomeReason:
    """Human-readable explanation attached to a WARN or BLOCK outcome."""
    code: str
    message: str