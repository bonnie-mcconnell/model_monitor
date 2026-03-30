from __future__ import annotations

import json
from typing import Any, Protocol, Sequence

import numpy as np
import jsonschema

from model_monitor.contracts.behavioral.evaluation import EvaluationResult
from model_monitor.utils.stats import cosine_similarity


class TextEncoder(Protocol):
    """
    Structural contract for any model that converts text to embedding vectors.

    Keeping this as a Protocol rather than an abstract base class means:
    - The real SentenceTransformer satisfies it without modification
    - Tests can inject a cheap stub without a 90MB model download
    - Swapping encoders (local vs API-based) requires no changes to the evaluator
    """

    def encode(self, sentences: list[str]) -> np.ndarray:
        """
        Encode a list of sentences into a 2-D array of shape (N, embedding_dim).
        """
        ...


class JsonValidityEvaluator:
    """
    Checks that the model output is parseable as JSON.

    The simplest possible evaluator: its purpose is to verify
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
        # Fail fast at construction - a bad schema is a programming error
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


class ToneConsistencyEvaluator:
    """
    Detects semantic drift between a model output and a set of reference outputs.

    Embeds the current output and all reference outputs using an injected encoder,
    computes cosine similarity between the current output's embedding and the
    centroid of the reference embeddings, and fails if similarity falls below
    the configured threshold.

    This is the first semantically meaningful evaluator in the contract system -
    it detects when an LLM's tone or style has drifted between model versions
    without requiring manual review.

    Design notes:
    - The encoder is injected (not constructed internally) so tests can pass a
      stub without downloading a model, and production can swap encoders without
      touching this class.
    - References are embedded once at construction time. The evaluator is
      intended to be constructed once per deployment, not per evaluation.
    - The centroid is recomputed on construction and cached. It does not change
      unless the evaluator is reconstructed with new references.
    """

    version = "1.0"

    def __init__(
        self,
        *,
        evaluator_id: str,
        encoder: TextEncoder,
        reference_outputs: Sequence[str],
        threshold: float = 0.75,
    ) -> None:
        if not reference_outputs:
            raise ValueError(
                "ToneConsistencyEvaluator requires at least one reference output"
            )
        if not (0.0 < threshold <= 1.0):
            raise ValueError(
                f"threshold must be in (0.0, 1.0], got {threshold}"
            )

        self.evaluator_id = evaluator_id
        self._encoder = encoder
        self._threshold = threshold

        # Embed all references and cache the centroid.
        # shape: (N, embedding_dim)
        reference_embeddings: np.ndarray = self._encoder.encode(
            list(reference_outputs)
        )
        # Elementwise mean across the N reference vectors → shape: (embedding_dim,)
        self._reference_centroid: np.ndarray = np.mean(
            reference_embeddings, axis=0
        )

    def evaluate(self, *, output: str) -> EvaluationResult:
        # shape: (1, embedding_dim) - encode expects a list
        embedding: np.ndarray = self._encoder.encode([output])[0]

        similarity = cosine_similarity(embedding, self._reference_centroid)

        if similarity < self._threshold:
            return EvaluationResult(
                passed=False,
                reason=(
                    f"tone drift detected: similarity {similarity:.2f}, "
                    f"threshold {self._threshold:.2f}"
                ),
            )

        return EvaluationResult(passed=True)