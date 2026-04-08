"""Append-only registry mapping evaluator IDs to evaluator instances."""
from __future__ import annotations

from model_monitor.contracts.evaluator import GuaranteeEvaluator


class EvaluatorRegistry:
    """
    Append-only registry mapping evaluator IDs to evaluator instances.

    Duplicate registration raises immediately - IDs are permanent.
    Unknown IDs raise at evaluation time, not silently pass.
    """
    def __init__(self) -> None:
        self._evaluators: dict[str, GuaranteeEvaluator] = {}

    def register(self, evaluator: GuaranteeEvaluator) -> None:
        if evaluator.evaluator_id in self._evaluators:
            raise ValueError(f"Evaluator already registered: {evaluator.evaluator_id}")
        self._evaluators[evaluator.evaluator_id] = evaluator

    def get(self, evaluator_id: str) -> GuaranteeEvaluator:
        try:
            return self._evaluators[evaluator_id]
        except KeyError:
            raise KeyError(f"Unknown evaluator: {evaluator_id}")