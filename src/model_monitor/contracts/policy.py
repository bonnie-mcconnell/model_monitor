from typing import Protocol, Iterable

from .behavioral.evaluation import GuaranteeEvaluation
from .outcome import DecisionOutcome, OutcomeReason


class DecisionPolicy(Protocol):
    policy_id: str

    def decide(
        self,
        *,
        guarantees: Iterable[GuaranteeEvaluation],
    ) -> tuple[DecisionOutcome, tuple[OutcomeReason, ...]]:
        ...
