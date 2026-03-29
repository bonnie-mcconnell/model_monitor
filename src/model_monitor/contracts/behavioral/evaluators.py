from __future__ import annotations

import json
from typing import Any

import jsonschema

from model_monitor.contracts.behavioral.evaluation import EvaluationResult


class JsonValidityEvaluator:
    """
    Checks that the model output is parseable as JSON.

    The simplest possible evaluator — its purpose is to verify
    the behavioral contract plumbing works end-to-end before adding
    semantic evaluators.
    """

    evaluator_id = "json_validity"
    version = "1.0"

    def evaluate(self, *, output: str) -> EvaluationResult:
        try:
            json.loads(output)
            return EvaluationResult(passed=True)
        except (json.JSONDecodeError, ValueError) as exc:
            return EvaluationResult(passed=False, reason=f"Invalid JSON: {exc}")


class JsonSchemaEvaluator:
    """
    Validates model output against a JSON Schema.

    The schema is bound at construction time so one evaluator instance
    corresponds to one schema version. This makes the registry
    append-only: new schema versions get new evaluator IDs rather than
    mutating existing ones.
    """

    version = "1.0"

    def __init__(self, *, evaluator_id: str, schema: dict[str, Any]) -> None:
        self.evaluator_id = evaluator_id
        self._schema = schema
        # Fail fast at construction — a bad schema is a programming error
        jsonschema.Draft7Validator.check_schema(schema)
        self._validator = jsonschema.Draft7Validator(schema)

    def evaluate(self, *, output: str) -> EvaluationResult:
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError) as exc:
            return EvaluationResult(passed=False, reason=f"Invalid JSON: {exc}")

        errors = list(self._validator.iter_errors(data))
        if not errors:
            return EvaluationResult(passed=True)

        first = errors[0]
        path = " → ".join(str(p) for p in first.absolute_path) or "root"
        return EvaluationResult(
            passed=False,
            reason=f"Schema violation at {path}: {first.message}",
        )