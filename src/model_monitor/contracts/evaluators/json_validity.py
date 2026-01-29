import json
from dataclasses import dataclass
from typing import Optional

from ..evaluator import EvaluationResult


@dataclass(frozen=True)
class _Result:
    passed: bool
    reason: Optional[str] = None


class JsonValidityEvaluator:
    evaluator_id = "json_validity"
    version = "1.0"

    def evaluate(self, *, output: str) -> EvaluationResult:
        try:
            json.loads(output)
            return _Result(passed=True)
        except Exception as e:
            return _Result(passed=False, reason=str(e))
