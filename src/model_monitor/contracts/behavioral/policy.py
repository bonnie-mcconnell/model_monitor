from __future__ import annotations

from typing import Iterable, Protocol

from model_monitor.contracts.behavioral.evaluation import GuaranteeEvaluation
from model_monitor.contracts.guarantee import Severity
from model_monitor.contracts.outcome import DecisionOutcome, OutcomeReason


class DecisionPolicy(Protocol):
    policy_id: str

    def decide(
        self,
        *,
        guarantees: Iterable[GuaranteeEvaluation],
    ) -> tuple[DecisionOutcome, tuple[OutcomeReason, ...]]: ...


class StrictBehaviorPolicy:
    """
    Blocks on any CRITICAL violation.
    Warns on two or more HIGH violations.
    Accepts otherwise.

    Severity comparison uses explicit equality rather than >= because
    Python Enum does not support ordering by default.
    """

    policy_id = "strict_behavior_v1"

    def decide(
        self,
        *,
        guarantees: Iterable[GuaranteeEvaluation],
    ) -> tuple[DecisionOutcome, tuple[OutcomeReason, ...]]:
        failures = [g for g in guarantees if not g.passed]

        if any(g.severity == Severity.CRITICAL for g in failures):
            return (
                DecisionOutcome.BLOCK,
                (OutcomeReason(
                    code="critical_violation",
                    message="Critical behavioral guarantee failed",
                ),),
            )

        if sum(1 for g in failures if g.severity == Severity.HIGH) >= 2:
            return (
                DecisionOutcome.WARN,
                (OutcomeReason(
                    code="multiple_high_failures",
                    message="Multiple high-severity behavioral regressions detected",
                ),),
            )

        return DecisionOutcome.ACCEPT, ()