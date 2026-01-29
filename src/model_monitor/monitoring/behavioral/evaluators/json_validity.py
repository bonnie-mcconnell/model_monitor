import json

from contracts.behavioral.evaluation import EvaluationResult


class JsonValidityEvaluator:
    evaluator_id = "json_validity"
    version = "1.0"

    def evaluate(self, *, output: str) -> EvaluationResult:
        try:
            json.loads(output)
            return EvaluationResult(passed=True)
        except Exception as e:
            return EvaluationResult(passed=False, reason=str(e))
