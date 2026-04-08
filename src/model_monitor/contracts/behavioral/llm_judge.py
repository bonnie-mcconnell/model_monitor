"""
LLMJudgeEvaluator - evaluates model outputs using an LLM as judge.

This evaluator sends a structured prompt to a language model asking it to
assess whether a candidate output is consistent with a set of reference
outputs, returning a structured verdict.

Architecture:
    The LLM client is injected via the LLMClient Protocol - the same
    dependency injection pattern used by ToneConsistencyEvaluator for its
    encoder. This means:

    - Tests inject a deterministic mock that runs instantly with no API key
    - Production injects a real client (Anthropic, OpenAI, etc.)
    - Swapping providers requires no changes to the evaluator class

Current status:
    The interface, prompt template, and output parsing are fully implemented.
    The AnthropicLLMClient below is a reference implementation - it requires
    an ANTHROPIC_API_KEY environment variable at construction time.
    Tests use MockLLMClient.

Prompt design:
    The prompt is structured to elicit a JSON verdict rather than free text.
    This avoids brittle regex parsing and makes the judge's reasoning
    inspectable alongside its verdict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from model_monitor.contracts.behavioral.evaluation import EvaluationResult

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """
    Structural contract for any LLM client used as a judge.

    The single method sends a prompt and returns the model's text response.
    Temperature and sampling parameters are the client's responsibility -
    the evaluator does not control them.
    """

    def complete(self, prompt: str) -> str:
        """
        Send prompt to the LLM and return the raw text response.
        """
        ...


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeVerdict:
    """
    Structured verdict produced by parsing the LLM judge's response.

    consistent: whether the candidate matches the reference style/tone
    score: confidence in the verdict, 0.0–1.0
    reasoning: the judge's explanation (useful for audit and debugging)
    """
    consistent: bool
    score: float
    reasoning: str

    @classmethod
    def from_json(cls, raw: str) -> JudgeVerdict:
        """
        Parse a JudgeVerdict from the LLM's JSON response.

        Raises ValueError if the response is malformed or missing fields.
        This is intentional: a judge that cannot produce structured output
        is a failing evaluator, not a silent pass.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM judge returned non-JSON response: {raw!r}"
            ) from exc

        missing = {"consistent", "score", "reasoning"} - data.keys()
        if missing:
            raise ValueError(
                f"LLM judge response missing fields: {missing}. Got: {raw!r}"
            )

        return cls(
            consistent=bool(data["consistent"]),
            score=float(data["score"]),
            reasoning=str(data["reasoning"]),
        )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """\
You are a behavioral regression evaluator for a production AI system.

Your task: assess whether the CANDIDATE OUTPUT is stylistically and tonally
consistent with the REFERENCE OUTPUTS. You are evaluating tone, register,
helpfulness, and communication style - not factual correctness.

REFERENCE OUTPUTS:
{references}

CANDIDATE OUTPUT:
{candidate}

Respond with a JSON object containing exactly these fields:
{{
  "consistent": true or false,
  "score": a float between 0.0 (completely inconsistent) and 1.0 (perfectly consistent),
  "reasoning": "one or two sentences explaining your verdict"
}}

Respond with JSON only. No preamble, no explanation outside the JSON object.
"""


def _build_prompt(
    candidate: str,
    references: list[str],
) -> str:
    formatted_refs = "\n\n".join(
        f"[Reference {i + 1}]\n{ref}" for i, ref in enumerate(references)
    )
    return _JUDGE_PROMPT_TEMPLATE.format(
        references=formatted_refs,
        candidate=candidate,
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class LLMJudgeEvaluator:
    """
    Evaluates model outputs using an LLM as judge.

    The judge receives a candidate output alongside reference outputs from
    a known-good model version and returns a structured consistency verdict.

    This evaluator complements ToneConsistencyEvaluator: where tone
    consistency uses embedding geometry (fast, deterministic, no API cost),
    the LLM judge uses natural language reasoning (slower, richer, catches
    subtle failures that embeddings miss - instruction adherence, safety
    posture, response completeness).

    Args:
        evaluator_id: unique identifier for registration in EvaluatorRegistry
        client: any LLMClient implementation
        reference_outputs: outputs from the reference model version
        consistency_threshold: minimum score to pass, in [0.0, 1.0]
    """

    version = "1.0"

    def __init__(
        self,
        *,
        evaluator_id: str,
        client: LLMClient,
        reference_outputs: list[str],
        consistency_threshold: float = 0.7,
    ) -> None:
        if not reference_outputs:
            raise ValueError(
                "LLMJudgeEvaluator requires at least one reference output"
            )
        if not (0.0 < consistency_threshold <= 1.0):
            raise ValueError(
                f"consistency_threshold must be in (0.0, 1.0], "
                f"got {consistency_threshold}"
            )

        self.evaluator_id = evaluator_id
        self._client = client
        self._references = list(reference_outputs)
        self._threshold = consistency_threshold

    def evaluate(self, *, output: str) -> EvaluationResult:
        prompt = _build_prompt(candidate=output, references=self._references)

        try:
            raw_response = self._client.complete(prompt)
            verdict = JudgeVerdict.from_json(raw_response)
        except ValueError as exc:
            # The judge produced unparseable output. Fail the guarantee with
            # a clear reason rather than letting a ValueError propagate up
            # through the contract runner and crash the evaluation.
            return EvaluationResult(
                passed=False,
                reason=f"LLM judge response unparseable: {exc}",
            )

        if not verdict.consistent or verdict.score < self._threshold:
            return EvaluationResult(
                passed=False,
                reason=(
                    f"LLM judge: inconsistent output "
                    f"(score {verdict.score:.2f}, threshold {self._threshold:.2f}). "
                    f"Reasoning: {verdict.reasoning}"
                ),
            )

        return EvaluationResult(passed=True)


# ---------------------------------------------------------------------------
# Mock implementation for tests
# ---------------------------------------------------------------------------

class MockLLMClient:
    """
    Deterministic LLM client for tests.

    Returns a pre-configured JSON verdict without any network call.
    Can be configured to return consistent or inconsistent verdicts,
    or to return malformed responses to test error handling.
    """

    def __init__(
        self,
        *,
        consistent: bool = True,
        score: float = 0.9,
        reasoning: str = "Outputs are stylistically consistent.",
        malformed: bool = False,
    ) -> None:
        self._consistent = consistent
        self._score = score
        self._reasoning = reasoning
        self._malformed = malformed
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)

        if self._malformed:
            return "this is not json at all"

        return json.dumps({
            "consistent": self._consistent,
            "score": self._score,
            "reasoning": self._reasoning,
        })


# ---------------------------------------------------------------------------
# Anthropic reference implementation (requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

class AnthropicLLMClient:
    """
    Production LLM client using the Anthropic API.

    Requires ANTHROPIC_API_KEY in the environment.
    Uses claude-haiku-4-5-20251001 by default - the fastest model, appropriate
    for high-frequency evaluation in a monitoring pipeline.

    Not imported in tests - the MockLLMClient covers all test scenarios.
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"
    MAX_TOKENS = 256

    def __init__(self, *, model: str = DEFAULT_MODEL) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required for AnthropicLLMClient. "
                "Install with: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self.MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(message.content[0].text)
