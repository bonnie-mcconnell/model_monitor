from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class EvaluationResult(Protocol):
    @property
    def passed(self) -> bool: ...

    @property
    def reason(self) -> Optional[str]: ...


@runtime_checkable
class GuaranteeEvaluator(Protocol):
    """
    Structural contract for all behavioral evaluators.

    @runtime_checkable allows isinstance() checks at registration time,
    which catches missing attributes before they surface as AttributeErrors
    mid-evaluation under production load.
    """

    evaluator_id: str
    version: str

    def evaluate(self, *, output: str) -> EvaluationResult: ...