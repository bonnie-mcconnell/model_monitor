from jsonschema import validate, ValidationError

from ..evaluation import EvaluationResult


class JsonSchemaEvaluator:
    evaluator_id = "json_schema_v1"
    version = "1.0"

    def __init__(self, schema: dict):
        self._schema = schema

    def evaluate(self, *, output: str) -> EvaluationResult:
        try:
            validate(instance=output, schema=self._schema)
            return EvaluationResult(passed=True)
        except ValidationError as e:
            return EvaluationResult(passed=False, reason=str(e))
