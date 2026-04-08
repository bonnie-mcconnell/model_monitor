"""
Tests for LLMJudgeEvaluator.

All tests use MockLLMClient - no API key required, no network calls,
runs in milliseconds. This is the same dependency injection pattern as
ToneConsistencyEvaluator's stub encoder.

The tests cover:
- Consistent verdict → pass
- Inconsistent verdict → fail with reason string
- Score below threshold → fail even when consistent=True
- Malformed LLM response → fail gracefully, no crash
- Prompt contains candidate and references
- Construction guards
"""
from __future__ import annotations

import json

import pytest

from model_monitor.contracts.behavioral.llm_judge import (
    JudgeVerdict,
    LLMJudgeEvaluator,
    MockLLMClient,
)
from model_monitor.contracts.evaluator import GuaranteeEvaluator

REFERENCE_OUTPUTS = [
    "Thank you for reaching out. I have reviewed your account and can confirm "
    "the refund has been processed.",
    "I appreciate your patience. The issue has been escalated to our technical "
    "team and they will follow up within 24 hours.",
]


# ---------------------------------------------------------------------------
# JudgeVerdict.from_json
# ---------------------------------------------------------------------------

def test_verdict_from_valid_json() -> None:
    raw = json.dumps({
        "consistent": True,
        "score": 0.85,
        "reasoning": "Outputs match in tone and register.",
    })
    verdict = JudgeVerdict.from_json(raw)
    assert verdict.consistent is True
    assert verdict.score == pytest.approx(0.85)
    assert "tone" in verdict.reasoning


def test_verdict_raises_on_non_json() -> None:
    with pytest.raises(ValueError, match="non-JSON"):
        JudgeVerdict.from_json("not json")


def test_verdict_raises_on_missing_fields() -> None:
    raw = json.dumps({"consistent": True})
    with pytest.raises(ValueError, match="missing fields"):
        JudgeVerdict.from_json(raw)


def test_verdict_is_frozen() -> None:
    verdict = JudgeVerdict(consistent=True, score=0.9, reasoning="ok")
    with pytest.raises((AttributeError, TypeError)):
        verdict.consistent = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMJudgeEvaluator - happy path
# ---------------------------------------------------------------------------

def test_passes_when_judge_returns_consistent() -> None:
    client = MockLLMClient(consistent=True, score=0.9)
    ev = LLMJudgeEvaluator(
        evaluator_id="llm_judge_v1",
        client=client,
        reference_outputs=REFERENCE_OUTPUTS,
        consistency_threshold=0.7,
    )
    result = ev.evaluate(output="Thank you for contacting us.")
    assert result.passed is True
    assert result.reason is None


def test_fails_when_judge_returns_inconsistent() -> None:
    client = MockLLMClient(
        consistent=False,
        score=0.3,
        reasoning="Tone is completely different - too terse and unhelpful.",
    )
    ev = LLMJudgeEvaluator(
        evaluator_id="llm_judge_v1",
        client=client,
        reference_outputs=REFERENCE_OUTPUTS,
    )
    result = ev.evaluate(output="Order shipped.")
    assert result.passed is False
    assert result.reason is not None


def test_fails_when_score_below_threshold_even_if_consistent() -> None:
    """
    consistent=True with score=0.5 below threshold=0.7 must still fail.
    The score is more informative than the boolean.
    """
    client = MockLLMClient(consistent=True, score=0.5)
    ev = LLMJudgeEvaluator(
        evaluator_id="llm_judge_v1",
        client=client,
        reference_outputs=REFERENCE_OUTPUTS,
        consistency_threshold=0.7,
    )
    result = ev.evaluate(output="Somewhat related response.")
    assert result.passed is False


def test_reason_string_contains_score_and_threshold() -> None:
    client = MockLLMClient(
        consistent=False,
        score=0.4,
        reasoning="Different register entirely.",
    )
    ev = LLMJudgeEvaluator(
        evaluator_id="llm_judge_v1",
        client=client,
        reference_outputs=REFERENCE_OUTPUTS,
        consistency_threshold=0.7,
    )
    result = ev.evaluate(output="Bad output.")
    assert result.reason is not None
    assert "0.40" in result.reason
    assert "0.70" in result.reason


def test_malformed_response_fails_gracefully() -> None:
    """
    If the LLM returns unparseable output, the evaluator must return a
    failed EvaluationResult with a clear reason - not raise an exception.
    An exception here would crash the BehavioralContractRunner and skip
    all subsequent evaluations in the pass.
    """
    client = MockLLMClient(malformed=True)
    ev = LLMJudgeEvaluator(
        evaluator_id="llm_judge_v1",
        client=client,
        reference_outputs=REFERENCE_OUTPUTS,
    )
    result = ev.evaluate(output="Any output.")
    assert result.passed is False
    assert result.reason is not None
    assert "unparseable" in result.reason


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_prompt_contains_candidate_output() -> None:
    """The LLM must see the candidate output in its prompt."""
    client = MockLLMClient()
    ev = LLMJudgeEvaluator(
        evaluator_id="llm_judge_v1",
        client=client,
        reference_outputs=REFERENCE_OUTPUTS,
    )
    ev.evaluate(output="This specific candidate text.")
    assert len(client.calls) == 1
    assert "This specific candidate text." in client.calls[0]


def test_prompt_contains_all_references() -> None:
    """Every reference output must appear in the prompt."""
    client = MockLLMClient()
    ev = LLMJudgeEvaluator(
        evaluator_id="llm_judge_v1",
        client=client,
        reference_outputs=REFERENCE_OUTPUTS,
    )
    ev.evaluate(output="output")
    prompt = client.calls[0]
    for ref in REFERENCE_OUTPUTS:
        assert ref in prompt


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

def test_raises_on_empty_references() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LLMJudgeEvaluator(
            evaluator_id="llm_judge_v1",
            client=MockLLMClient(),
            reference_outputs=[],
        )


def test_raises_on_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        LLMJudgeEvaluator(
            evaluator_id="llm_judge_v1",
            client=MockLLMClient(),
            reference_outputs=REFERENCE_OUTPUTS,
            consistency_threshold=0.0,
        )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_evaluator_satisfies_guarantee_evaluator_protocol() -> None:
    ev = LLMJudgeEvaluator(
        evaluator_id="llm_judge_v1",
        client=MockLLMClient(),
        reference_outputs=REFERENCE_OUTPUTS,
    )
    assert isinstance(ev, GuaranteeEvaluator)
