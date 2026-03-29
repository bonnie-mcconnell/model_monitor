from typing import Protocol, Optional, runtime_checkable


@runtime_checkable
class EvaluationResult(Protocol):
    @property
    def passed(self) -> bool: ...

    @property
    def reason(self) -> Optional[str]: ...


class GuaranteeEvaluator(Protocol):
    evaluator_id: str
    version: str

    def evaluate(self, *, output: str) -> EvaluationResult: ...