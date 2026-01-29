from typing import Protocol, Optional


class EvaluationResult(Protocol):
    passed: bool
    reason: Optional[str]


class GuaranteeEvaluator(Protocol):
    evaluator_id: str
    version: str

    def evaluate(self, *, output: str) -> EvaluationResult:
        ...
