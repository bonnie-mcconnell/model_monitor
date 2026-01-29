from typing import Dict

from .evaluator import GuaranteeEvaluator


class EvaluatorRegistry:
    def __init__(self) -> None:
        self._evaluators: Dict[str, GuaranteeEvaluator] = {}

    def register(self, evaluator: GuaranteeEvaluator) -> None:
        if evaluator.evaluator_id in self._evaluators:
            raise ValueError(f"Evaluator already registered: {evaluator.evaluator_id}")
        self._evaluators[evaluator.evaluator_id] = evaluator

    def get(self, evaluator_id: str) -> GuaranteeEvaluator:
        try:
            return self._evaluators[evaluator_id]
        except KeyError:
            raise KeyError(f"Unknown evaluator: {evaluator_id}")
