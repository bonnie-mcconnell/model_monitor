"""
Integration test for the full behavioral contracts pipeline.

Tests the complete path:
  load_contract_from_yaml → EvaluatorRegistry → BehavioralContractRunner
  → StrictBehaviorPolicy → DecisionRecord

This is the only test that exercises the system the way it runs in
production and in the demo script: contract loaded from YAML, evaluators
registered, runner evaluating real model outputs.

Uses only JsonValidityEvaluator and JsonSchemaEvaluator so no
sentence-transformers download is required. ToneConsistencyEvaluator
integration is covered by test_tone_consistency_evaluator.py.
"""
from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path

import pytest

from model_monitor.contracts.behavioral.context import DecisionContext
from model_monitor.contracts.behavioral.evaluators import (
    JsonSchemaEvaluator,
    JsonValidityEvaluator,
)
from model_monitor.contracts.behavioral.policy import StrictBehaviorPolicy
from model_monitor.contracts.behavioral.runner import BehavioralContractRunner
from model_monitor.contracts.loader import load_contract_from_yaml
from model_monitor.contracts.outcome import DecisionOutcome
from model_monitor.contracts.registry import EvaluatorRegistry
from model_monitor.contracts.validator import validate_contract

SUPPORT_SCHEMA = {
    "type": "object",
    "required": ["ticket_id", "response"],
    "properties": {
        "ticket_id": {"type": "string"},
        "response": {"type": "string"},
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def contract_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "test_contract.yaml"
    path.write_text(textwrap.dedent("""
        contract_id: support_v1
        version: "1.0"
        scope: support_responses

        guarantees:
          - id: valid_json
            description: Must be valid JSON
            severity: critical
            evaluator: json_validity

          - id: schema_check
            description: Must match support response schema
            severity: critical
            evaluator: json_schema_support
    """))
    return path


@pytest.fixture
def registry() -> EvaluatorRegistry:
    reg = EvaluatorRegistry()
    reg.register(JsonValidityEvaluator())
    reg.register(JsonSchemaEvaluator(
        evaluator_id="json_schema_support",
        schema=SUPPORT_SCHEMA,
    ))
    return reg


@pytest.fixture
def runner(contract_yaml: Path, registry: EvaluatorRegistry) -> BehavioralContractRunner:
    contract = load_contract_from_yaml(str(contract_yaml))
    validate_contract(contract, registry)
    return BehavioralContractRunner(
        contract=contract,
        registry=registry,
        policy=StrictBehaviorPolicy(),
    )


def _context(output: str) -> DecisionContext:
    return DecisionContext(
        run_id=str(uuid.uuid4()),
        model_id="test-model-v1",
        prompt_id="p1",
        output=output,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_valid_output_produces_accept(runner: BehavioralContractRunner) -> None:
    output = json.dumps({"ticket_id": "T-001", "response": "Thank you."})
    record = runner.evaluate(_context(output))
    assert record.outcome == DecisionOutcome.ACCEPT


def test_invalid_json_produces_block(runner: BehavioralContractRunner) -> None:
    record = runner.evaluate(_context("not json at all"))
    assert record.outcome == DecisionOutcome.BLOCK


def test_schema_violation_produces_block(runner: BehavioralContractRunner) -> None:
    output = json.dumps({"wrong_field": "value"})
    record = runner.evaluate(_context(output))
    assert record.outcome == DecisionOutcome.BLOCK


def test_decision_record_has_full_provenance(runner: BehavioralContractRunner) -> None:
    """
    Every DecisionRecord must carry enough information to reconstruct
    what happened without re-running the evaluation.
    """
    output = json.dumps({"ticket_id": "T-001", "response": "Hello."})
    record = runner.evaluate(_context(output))

    assert record.decision_id  # non-empty UUID
    assert record.context.model_id == "test-model-v1"
    assert record.created_at is not None
    assert len(record.guarantees) == 2  # one per guarantee in contract


def test_each_guarantee_has_evaluator_provenance(
    runner: BehavioralContractRunner,
) -> None:
    output = json.dumps({"ticket_id": "T-001", "response": "Hello."})
    record = runner.evaluate(_context(output))

    for g in record.guarantees:
        assert g.evaluator_id  # must be set
        assert g.evaluator_version  # must be set
        assert g.severity is not None


def test_failed_guarantee_reason_is_populated(
    runner: BehavioralContractRunner,
) -> None:
    record = runner.evaluate(_context("not json"))

    failed = [g for g in record.guarantees if not g.passed]
    assert len(failed) > 0
    for g in failed:
        assert g.reason is not None and len(g.reason) > 0


def test_decision_record_is_immutable(runner: BehavioralContractRunner) -> None:
    output = json.dumps({"ticket_id": "T-001", "response": "Hello."})
    record = runner.evaluate(_context(output))

    with pytest.raises((AttributeError, TypeError)):
        record.outcome = DecisionOutcome.BLOCK  # type: ignore[misc]


def test_consecutive_evaluations_produce_unique_decision_ids(
    runner: BehavioralContractRunner,
) -> None:
    """Each evaluation must have a unique decision_id for audit deduplication."""
    output = json.dumps({"ticket_id": "T-001", "response": "Hello."})
    ids = {runner.evaluate(_context(output)).decision_id for _ in range(5)}
    assert len(ids) == 5


def test_validate_contract_passes_before_run(
    contract_yaml: Path,
    registry: EvaluatorRegistry,
) -> None:
    """validate_contract must not raise when all evaluators are registered."""
    contract = load_contract_from_yaml(str(contract_yaml))
    validate_contract(contract, registry)  # must not raise


def test_real_support_response_yaml_loads_and_validates() -> None:
    """
    Smoke-test the actual example contract in contracts/examples/.
    This confirms the file is valid YAML and references evaluators
    that can be registered.
    """
    yaml_path = Path(__file__).parent.parent / "contracts" / "examples" / "support_response.yaml"
    if not yaml_path.exists():
        pytest.skip("contracts/examples/support_response.yaml not present")

    contract = load_contract_from_yaml(str(yaml_path))

    assert contract.contract_id == "support_response_v1"
    assert len(contract.guarantees) == 3

    # Verify the evaluator IDs match what the demo script registers
    evaluator_ids = {g.evaluator_id for g in contract.guarantees}
    assert "json_validity" in evaluator_ids
    assert "json_schema_support_v1" in evaluator_ids
    assert "tone_consistency_support_v1" in evaluator_ids
