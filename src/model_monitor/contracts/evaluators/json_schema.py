import json
from dataclasses import dataclass
from typing import Optional

from jsonschema import validate, ValidationError

from ..evaluator import EvaluationResult


@dataclass(frozen=True)
class _Result:
    passed: bool
    reason: Optional[str] = None


class JsonSchemaEvaluator:
    evaluator_id = "json_schema_v1"
    version = "1.0"

    def __init__(self, *, schema: dict) -> None:
        self._schema = schema

    def evaluate(self, *, output: str) -> EvaluationResult:
        try:
            data = json.loads(output)
            validate(instance=data, schema=self._schema)
            return _Result(passed=True)
        except (json.JSONDecodeError, ValidationError) as e:
            return _Result(passed=False, reason=str(e))
