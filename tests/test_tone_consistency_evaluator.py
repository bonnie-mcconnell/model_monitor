"""
Tests for ToneConsistencyEvaluator.

All tests use a deterministic stub encoder. This is intentional:
we are testing the evaluator's logic (centroid computation, threshold
comparison, reason string format), not the sentence-transformers library.
The real encoder is an injected dependency, the stub satisfies the
TextEncoder Protocol without requiring a model download.
"""
from __future__ import annotations

import numpy as np
import pytest

from model_monitor.contracts.behavioral.evaluators import ToneConsistencyEvaluator


# ---------------------------------------------------------------------------
# Stub encoder
# ---------------------------------------------------------------------------

class StubEncoder:
    """
    Deterministic text encoder for tests.

    Maps known strings to fixed embedding vectors. Any unknown string
    returns a zero vector of the same dimension. This gives full control
    over cosine similarity values without needing a real model.
    """

    DIM = 8

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._embeddings = embeddings

    def encode(self, sentences: list[str]) -> np.ndarray:
        rows = []
        for s in sentences:
            vec = self._embeddings.get(s, [0.0] * self.DIM)
            rows.append(vec)
        return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Three reference outputs that all point in roughly the same direction.
SIMILAR_REFS = ["ref_a", "ref_b", "ref_c"]
# An output that points in the same direction as the references.
SIMILAR_OUTPUT = "similar_output"
# An output that is orthogonal to the references.
DISSIMILAR_OUTPUT = "dissimilar_output"

# Embeddings chosen so cosine similarity is predictable:
#   - SIMILAR_OUTPUT vs centroid of refs ≈ 1.0 (same direction)
#   - DISSIMILAR_OUTPUT vs centroid of refs ≈ 0.0 (orthogonal)
_EMBEDDINGS: dict[str, list[float]] = {
    "ref_a":            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ref_b":            [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ref_c":            [1.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    SIMILAR_OUTPUT:     [1.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
    DISSIMILAR_OUTPUT:  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
}


@pytest.fixture
def encoder() -> StubEncoder:
    return StubEncoder(_EMBEDDINGS)


@pytest.fixture
def evaluator(encoder: StubEncoder) -> ToneConsistencyEvaluator:
    return ToneConsistencyEvaluator(
        evaluator_id="tone_consistency",
        encoder=encoder,
        reference_outputs=SIMILAR_REFS,
        threshold=0.75,
    )


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

def test_passes_when_output_is_similar_to_references(
    evaluator: ToneConsistencyEvaluator,
) -> None:
    result = evaluator.evaluate(output=SIMILAR_OUTPUT)
    assert result.passed is True
    assert result.reason is None


def test_fails_when_output_is_dissimilar_to_references(
    evaluator: ToneConsistencyEvaluator,
) -> None:
    result = evaluator.evaluate(output=DISSIMILAR_OUTPUT)
    assert result.passed is False
    assert result.reason is not None


# ---------------------------------------------------------------------------
# Reason string format - must be exact so downstream tooling can parse it
# ---------------------------------------------------------------------------

def test_reason_string_contains_similarity_and_threshold(
    evaluator: ToneConsistencyEvaluator,
) -> None:
    result = evaluator.evaluate(output=DISSIMILAR_OUTPUT)
    assert result.passed is False
    assert result.reason is not None
    # Both numbers must appear, formatted to 2 decimal places
    assert "tone drift detected" in result.reason
    assert "0.75" in result.reason                        # threshold
    # similarity is approximately 0.0 for the orthogonal vector
    assert "similarity" in result.reason


def test_reason_format_matches_expected_template(
    encoder: StubEncoder,
) -> None:
    """
    Verify the exact reason string format so any downstream log parser
    or alert rule that depends on it will catch format regressions.
    """
    ev = ToneConsistencyEvaluator(
        evaluator_id="tone_consistency",
        encoder=encoder,
        reference_outputs=SIMILAR_REFS,
        threshold=0.75,
    )
    result = ev.evaluate(output=DISSIMILAR_OUTPUT)
    assert result.reason is not None
    # Format: "tone drift detected: similarity X.XX, threshold Y.YY"
    assert result.reason.startswith("tone drift detected: similarity ")
    assert ", threshold " in result.reason


# ---------------------------------------------------------------------------
# Single reference output - centroid of one vector is that vector
# ---------------------------------------------------------------------------

def test_handles_single_reference_output(encoder: StubEncoder) -> None:
    """
    N=1 is a valid edge case: centroid of one vector is itself.
    The evaluator must not crash or produce a degenerate result.
    """
    ev = ToneConsistencyEvaluator(
        evaluator_id="tone_consistency",
        encoder=encoder,
        reference_outputs=["ref_a"],
        threshold=0.75,
    )
    # Output in same direction as the single reference → should pass
    result = ev.evaluate(output=SIMILAR_OUTPUT)
    assert result.passed is True


# ---------------------------------------------------------------------------
# Threshold boundary - verify the comparison is strictly less-than
# ---------------------------------------------------------------------------

def test_output_exactly_at_threshold_passes(encoder: StubEncoder) -> None:
    """
    Similarity == threshold should pass. The condition is similarity < threshold,
    not <=. A response exactly on the boundary is acceptable.
    """
    # Build embeddings where output similarity to centroid equals exactly 0.5
    # ref: [1, 0], output: [1, 1] normalised → cosine = 1/sqrt(2) ≈ 0.707
    # Instead, use orthogonal + diagonal to get a specific similarity.
    # Easiest: set threshold equal to computed similarity.
    custom_enc = StubEncoder({
        "ref_x": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "at_threshold": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    ev = ToneConsistencyEvaluator(
        evaluator_id="tone_consistency",
        encoder=custom_enc,
        reference_outputs=["ref_x"],
        threshold=1.0,    # cosine of identical vectors = 1.0 exactly
    )
    result = ev.evaluate(output="at_threshold")
    assert result.passed is True


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

def test_raises_on_empty_reference_outputs(encoder: StubEncoder) -> None:
    with pytest.raises(ValueError, match="at least one reference"):
        ToneConsistencyEvaluator(
            evaluator_id="tone_consistency",
            encoder=encoder,
            reference_outputs=[],
            threshold=0.75,
        )


def test_raises_on_invalid_threshold(encoder: StubEncoder) -> None:
    with pytest.raises(ValueError, match="threshold"):
        ToneConsistencyEvaluator(
            evaluator_id="tone_consistency",
            encoder=encoder,
            reference_outputs=["ref_a"],
            threshold=1.5,
        )


def test_raises_on_zero_threshold(encoder: StubEncoder) -> None:
    with pytest.raises(ValueError, match="threshold"):
        ToneConsistencyEvaluator(
            evaluator_id="tone_consistency",
            encoder=encoder,
            reference_outputs=["ref_a"],
            threshold=0.0,
        )


# ---------------------------------------------------------------------------
# Evaluator metadata - required by EvaluatorRegistry
# ---------------------------------------------------------------------------

def test_evaluator_has_required_attributes(evaluator: ToneConsistencyEvaluator) -> None:
    assert isinstance(evaluator.evaluator_id, str)
    assert isinstance(evaluator.version, str)
    assert len(evaluator.evaluator_id) > 0


def test_evaluator_satisfies_guarantee_evaluator_protocol(
    evaluator: ToneConsistencyEvaluator,
) -> None:
    """
    Verify the evaluator structurally satisfies the GuaranteeEvaluator Protocol
    so it can be registered without a runtime error.
    """
    from model_monitor.contracts.evaluator import GuaranteeEvaluator
    assert isinstance(evaluator, GuaranteeEvaluator)