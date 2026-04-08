"""GuaranteeEvaluator Protocol - structural contract for all evaluators."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from model_monitor.contracts.behavioral.evaluation import EvaluationResult


@runtime_checkable
class GuaranteeEvaluator(Protocol):
    """
    Structural contract for all behavioral evaluators.

    Any class with ``evaluator_id: str``, ``version: str``, and an
    ``evaluate(*, output: str) -> EvaluationResult`` method satisfies this
    Protocol - no inheritance required.

    ``@runtime_checkable`` enables ``isinstance(evaluator, GuaranteeEvaluator)``
    checks at registry registration time, which catches missing attributes
    before they surface as AttributeErrors mid-evaluation under production load.
    """

    evaluator_id: str
    version: str

    def evaluate(self, *, output: str) -> EvaluationResult: ...
