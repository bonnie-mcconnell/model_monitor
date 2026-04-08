"""
Latency benchmark for behavioral contract evaluators.

Measures P50/P95/P99 wall-clock time per evaluator over a configurable
number of iterations so the per-request cost of running behavioral
contracts inside predict_batch is concrete rather than estimated.

These numbers directly inform the latency budget decision documented in
the architecture: whether ToneConsistencyEvaluator (~10ms on CPU) is
acceptable on the critical path depends on the SLA of the inference
endpoint, and this script measures it under the same conditions.

Usage:
    python scripts/bench_evaluators.py
    python scripts/bench_evaluators.py --iterations 500 --warmup 20

Output:
    evaluator                     p50      p95      p99    total
    ────────────────────────────────────────────────────────────
    JsonValidityEvaluator        0.01ms   0.02ms   0.03ms  100
    JsonSchemaEvaluator          0.05ms   0.08ms   0.12ms  100
    ToneConsistencyEvaluator    11.20ms  13.40ms  15.10ms  100
    LLMJudgeEvaluator (mock)     0.02ms   0.03ms   0.04ms  100
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Protocol

import numpy as np

from model_monitor.contracts.behavioral.evaluators import (
    JsonSchemaEvaluator,
    JsonValidityEvaluator,
    ToneConsistencyEvaluator,
)
from model_monitor.contracts.behavioral.llm_judge import (
    LLMJudgeEvaluator,
    MockLLMClient,
)

# ---------------------------------------------------------------------------
# Stub encoder - same deterministic implementation used in tests, so results
# are reproducible and do not require a model download.  Replace with the
# real SentenceTransformer to measure production latency.
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 384


class _StubEncoder:
    """Fast deterministic encoder for benchmarking the evaluation path."""

    def encode(self, sentences: list[str]) -> np.ndarray:
        rng = np.random.default_rng(seed=sum(len(s) for s in sentences))
        raw = rng.standard_normal((len(sentences), _EMBEDDING_DIM)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return (raw / np.where(norms == 0, 1.0, norms)).astype(np.float32)


# ---------------------------------------------------------------------------
# Protocol for anything with an evaluate() method
# ---------------------------------------------------------------------------

class _HasEvaluate(Protocol):
    def evaluate(self, *, output: str) -> Any: ...


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

_VALID_JSON_OUTPUT = '{"ticket_id": "T-001", "response": "Thank you for reaching out."}'

_SUPPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["ticket_id", "response"],
    "properties": {
        "ticket_id": {"type": "string"},
        "response": {"type": "string"},
    },
    "additionalProperties": False,
}

_REFERENCE_OUTPUTS = [
    '{"ticket_id": "T-001", "response": "Thank you for contacting support."}',
    '{"ticket_id": "T-002", "response": "I have reviewed your account and will resolve this."}',
    '{"ticket_id": "T-003", "response": "We appreciate your patience and will follow up shortly."}',
]


def _build_evaluators() -> list[tuple[str, _HasEvaluate]]:
    encoder = _StubEncoder()
    return [
        (
            "JsonValidityEvaluator",
            JsonValidityEvaluator(),
        ),
        (
            "JsonSchemaEvaluator",
            JsonSchemaEvaluator(
                evaluator_id="json_schema_support_v1",
                schema=_SUPPORT_SCHEMA,
            ),
        ),
        (
            "ToneConsistencyEvaluator",
            ToneConsistencyEvaluator(
                evaluator_id="tone_consistency_support_v1",
                encoder=encoder,
                reference_outputs=_REFERENCE_OUTPUTS,
                threshold=0.75,
            ),
        ),
        (
            "LLMJudgeEvaluator (mock)",
            LLMJudgeEvaluator(
                evaluator_id="llm_judge_v1",
                client=MockLLMClient(consistent=True, score=0.9),
                reference_outputs=_REFERENCE_OUTPUTS,
            ),
        ),
    ]


def _bench_one(
    evaluator: _HasEvaluate,
    output: str,
    *,
    n: int,
    warmup: int,
) -> list[float]:
    """Return a list of wall-clock durations in milliseconds."""
    # Warmup - JIT compilation, cache fills, branch predictor warm-up
    for _ in range(warmup):
        evaluator.evaluate(output=output)

    durations: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        evaluator.evaluate(output=output)
        durations.append((time.perf_counter() - t0) * 1000)

    return durations


def _percentile(data: list[float], pct: float) -> float:
    """Return the given percentile of data using linear interpolation."""
    return float(statistics.quantiles(data, n=100, method="inclusive")[int(pct) - 1])


def _print_results(results: list[tuple[str, list[float]]]) -> None:
    col_w = max(len(name) for name, _ in results) + 2
    header = f"  {'evaluator':<{col_w}} {'p50':>8}  {'p95':>8}  {'p99':>8}  {'n':>6}"
    rule = "  " + "─" * (col_w + 36)

    print(f"\n{header}")
    print(rule)

    for name, durations in results:
        p50 = _percentile(durations, 50)
        p95 = _percentile(durations, 95)
        p99 = _percentile(durations, 99)
        n = len(durations)
        print(f"  {name:<{col_w}} {p50:>6.2f}ms  {p95:>6.2f}ms  {p99:>6.2f}ms  {n:>6}")

    print(rule)
    print(
        "\n  Note: ToneConsistencyEvaluator uses a stub encoder (no model download)."
        "\n  Run with --real-encoder to load all-MiniLM-L6-v2 and measure"
        "\n  production latency.\n"
    )


def _load_real_encoder() -> Any | None:  # pragma: no cover
    """
    Load the production SentenceTransformer encoder.

    Returns None and prints a warning when sentence-transformers is not installed,
    so --real-encoder degrades gracefully instead of crashing.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        print("sentence-transformers not installed - using stub encoder.\n")
        return None


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Benchmark behavioral contract evaluator latency."
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of timed iterations per evaluator (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before timing begins (default: 10)",
    )
    parser.add_argument(
        "--real-encoder",
        action="store_true",
        help="Load all-MiniLM-L6-v2 for ToneConsistencyEvaluator (requires sentence-transformers)",
    )
    args = parser.parse_args()

    evaluators = _build_evaluators()

    if args.real_encoder:
        _encoder = _load_real_encoder()
        if _encoder is not None:
            evaluators[2] = (
                "ToneConsistencyEvaluator (real)",
                ToneConsistencyEvaluator(
                    evaluator_id="tone_consistency_real",
                    encoder=_encoder,
                    reference_outputs=_REFERENCE_OUTPUTS,
                    threshold=0.75,
                ),
            )

    print(f"\nBenchmarking {args.iterations} iterations "
          f"(+{args.warmup} warmup) per evaluator ...")

    results: list[tuple[str, list[float]]] = []
    for name, ev in evaluators:
        durations = _bench_one(
            ev,
            _VALID_JSON_OUTPUT,
            n=args.iterations,
            warmup=args.warmup,
        )
        results.append((name, durations))

    _print_results(results)


if __name__ == "__main__":  # pragma: no cover
    main()
