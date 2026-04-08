"""
End-to-end demonstration of the behavioral contracts system.

Loads the support_response contract from YAML, builds an EvaluatorRegistry
with all three evaluators, runs four hardcoded model outputs through
BehavioralContractRunner, and prints each DecisionRecord clearly.

Usage:
    python scripts/demo_contracts.py

No external services required. sentence-transformers runs locally on CPU.
First run downloads the all-MiniLM-L6-v2 model (~90MB) and caches it.
Subsequent runs are instant.
"""
from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path

from model_monitor.contracts.behavioral.context import DecisionContext
from model_monitor.contracts.behavioral.evaluators import (
    JsonSchemaEvaluator,
    JsonValidityEvaluator,
    ToneConsistencyEvaluator,
)
from model_monitor.contracts.behavioral.policy import StrictBehaviorPolicy
from model_monitor.contracts.behavioral.records import DecisionRecord
from model_monitor.contracts.behavioral.runner import BehavioralContractRunner
from model_monitor.contracts.loader import load_contract_from_yaml
from model_monitor.contracts.outcome import DecisionOutcome
from model_monitor.contracts.registry import EvaluatorRegistry


# ---------------------------------------------------------------------------
# Schema for the support response format
# ---------------------------------------------------------------------------

SUPPORT_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "required": ["ticket_id", "response"],
    "properties": {
        "ticket_id": {"type": "string"},
        "response": {"type": "string"},
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Reference outputs - what good support responses look like.
# Embedded once at construction time; centroid cached inside the evaluator.
# ---------------------------------------------------------------------------

REFERENCE_OUTPUTS: list[str] = [
    json.dumps({
        "ticket_id": "T-001",
        "response": (
            "Thank you for reaching out. I have reviewed your account and "
            "can confirm that the refund has been processed. You should see "
            "the amount credited within 3–5 business days. Please let me "
            "know if there is anything else I can help you with."
        ),
    }),
    json.dumps({
        "ticket_id": "T-002",
        "response": (
            "I appreciate your patience while we looked into this. The issue "
            "has been escalated to our technical team and they will follow up "
            "with you directly within 24 hours. We apologise for the "
            "inconvenience caused."
        ),
    }),
    json.dumps({
        "ticket_id": "T-003",
        "response": (
            "Thank you for contacting us. I have updated your subscription "
            "plan as requested and the changes will take effect at the start "
            "of your next billing cycle. A confirmation email has been sent "
            "to your registered address."
        ),
    }),
]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

GOOD_RESPONSE = json.dumps({
    "ticket_id": "T-100",
    "response": (
        "Thank you for getting in touch. I have located your order and can "
        "confirm it has been dispatched. You should receive it within 2 "
        "business days. Please do not hesitate to contact us if you need "
        "further assistance."
    ),
})

# Correct structure but extremely terse - tone drift
TERSE_RESPONSE = json.dumps({
    "ticket_id": "T-101",
    "response": "Order shipped. Arrives in 2 days.",
})

# Valid JSON but missing required 'response' field
MISSING_FIELD_RESPONSE = json.dumps({
    "ticket_id": "T-102",
    "message": "This field name is wrong.",
})

# Not JSON at all
NOT_JSON_RESPONSE = (
    "Your order has shipped and will arrive in 2 business days."
)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_registry() -> EvaluatorRegistry:
    """
    Construct and return a populated EvaluatorRegistry.

    SentenceTransformer is instantiated here, not at module import time,
    so the model is only loaded when the demo is explicitly run.
    """
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Encoder: all-MiniLM-L6-v2\n")
    except ImportError:
        print("  sentence-transformers not installed. Run: pip install sentence-transformers\n")
        raise

    registry = EvaluatorRegistry()
    registry.register(JsonValidityEvaluator())
    registry.register(
        JsonSchemaEvaluator(
            evaluator_id="json_schema_support_v1",
            schema=SUPPORT_RESPONSE_SCHEMA,
        )
    )
    registry.register(
        ToneConsistencyEvaluator(
            evaluator_id="tone_consistency_support_v1",
            encoder=encoder,
            reference_outputs=REFERENCE_OUTPUTS,
            threshold=0.60,
        )
    )
    return registry


def make_context(output: str, label: str) -> DecisionContext:
    return DecisionContext(
        run_id=str(uuid.uuid4()),
        model_id="support-llm-v2",
        prompt_id=label,
        output=output,
        metadata={"demo": True},
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_OUTCOME_SYMBOL = {
    DecisionOutcome.ACCEPT: "✓  ACCEPT",
    DecisionOutcome.WARN:   "⚠  WARN  ",
    DecisionOutcome.BLOCK:  "✗  BLOCK ",
}

_COLOUR = {
    DecisionOutcome.ACCEPT: "\033[32m",
    DecisionOutcome.WARN:   "\033[33m",
    DecisionOutcome.BLOCK:  "\033[31m",
}
_RESET = "\033[0m"


def print_record(label: str, record: DecisionRecord) -> None:
    colour = _COLOUR.get(record.outcome, "")
    symbol = _OUTCOME_SYMBOL.get(record.outcome, str(record.outcome))

    print(f"  {colour}{symbol}{_RESET}  [{label}]")
    print(f"           decision_id : {record.decision_id}")
    print(f"           evaluated   : {record.created_at.strftime('%H:%M:%S UTC')}")

    for g in record.guarantees:
        status = "pass" if g.passed else f"FAIL ({g.severity.value})"
        print(f"           {g.evaluator_id:<36} {status}")
        if g.reason:
            wrapped = textwrap.fill(
                g.reason,
                width=60,
                initial_indent=" " * 49,
                subsequent_indent=" " * 49,
            )
            print(wrapped)

    for r in record.reasons:
        print(f"           → {r.code}: {r.message}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    contract_path = (
        Path(__file__).parent.parent / "contracts" / "examples" / "support_response.yaml"
    )

    print("\n━━━  model_monitor behavioral contracts demo  ━━━\n")
    print("  Loading encoder (cached after first run)...")

    registry = build_registry()

    contract = load_contract_from_yaml(str(contract_path))
    print(f"  Contract : {contract.contract_id}  v{contract.version}")
    print(f"  Scope    : {contract.scope}")
    print(f"  Checks   : {len(contract.guarantees)} guarantees per output")
    print()
    print("─" * 60)
    print()

    runner = BehavioralContractRunner(
        contract=contract,
        registry=registry,
        policy=StrictBehaviorPolicy(),
    )

    cases: list[tuple[str, str]] = [
        ("good response",    GOOD_RESPONSE),
        ("terse response",   TERSE_RESPONSE),
        ("missing field",    MISSING_FIELD_RESPONSE),
        ("not JSON",         NOT_JSON_RESPONSE),
    ]

    for label, output in cases:
        record = runner.evaluate(make_context(output, label))
        print_record(label, record)

    print("━" * 60)
    print()


if __name__ == "__main__":
    main()
