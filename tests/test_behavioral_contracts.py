import pytest
import json

from model_monitor.contracts.behavioral.evaluators import (
    JsonValidityEvaluator,
    JsonSchemaEvaluator,
)
from model_monitor.contracts.behavioral.policy import StrictBehaviorPolicy
from model_monitor.contracts.behavioral.evaluation import GuaranteeEvaluation
from model_monitor.contracts.guarantee import Severity
from model_monitor.contracts.outcome import DecisionOutcome


# ── JsonValidityEvaluator ─────────────────────────────────────────────────────

def test_json_validity_passes_on_valid_json():
    ev = JsonValidityEvaluator()
    result = ev.evaluate(output='{"key": "value"}')
    assert result.passed is True
    assert result.reason is None


def test_json_validity_fails_on_invalid_json():
    ev = JsonValidityEvaluator()
    result = ev.evaluate(output="not json at all")
    assert result.passed is False
    assert result.reason is not None
    assert "JSON" in result.reason


def test_json_validity_fails_on_empty_string():
    ev = JsonValidityEvaluator()
    result = ev.evaluate(output="")
    assert result.passed is False


# ── JsonSchemaEvaluator ───────────────────────────────────────────────────────

SUPPORT_SCHEMA = {
    "type": "object",
    "required": ["ticket_id", "response"],
    "properties": {
        "ticket_id": {"type": "string"},
        "response": {"type": "string"},
    },
    "additionalProperties": False,
}


def test_json_schema_passes_on_conforming_output():
    ev = JsonSchemaEvaluator(
        evaluator_id="json_schema_support_v1",
        schema=SUPPORT_SCHEMA,
    )
    output = json.dumps({"ticket_id": "T-123", "response": "Hello"})
    result = ev.evaluate(output=output)
    assert result.passed is True


def test_json_schema_fails_on_missing_required_field():
    ev = JsonSchemaEvaluator(
        evaluator_id="json_schema_support_v1",
        schema=SUPPORT_SCHEMA,
    )
    output = json.dumps({"ticket_id": "T-123"})  # missing 'response'
    result = ev.evaluate(output=output)
    assert result.passed is False
    assert result.reason is not None and "response" in result.reason


def test_json_schema_fails_on_wrong_type():
    ev = JsonSchemaEvaluator(
        evaluator_id="json_schema_support_v1",
        schema=SUPPORT_SCHEMA,
    )
    output = json.dumps({"ticket_id": 999, "response": "Hello"})  # int not string
    result = ev.evaluate(output=output)
    assert result.passed is False


def test_json_schema_fails_gracefully_on_invalid_json():
    ev = JsonSchemaEvaluator(
        evaluator_id="json_schema_support_v1",
        schema=SUPPORT_SCHEMA,
    )
    result = ev.evaluate(output="{broken")
    assert result.passed is False
    assert result.reason is not None


def test_json_schema_rejects_bad_schema_at_construction():
    with pytest.raises(Exception):
        JsonSchemaEvaluator(
            evaluator_id="bad",
            schema={"type": "not_a_real_type"},
        )


# ── StrictBehaviorPolicy ──────────────────────────────────────────────────────

def _make_evaluation(passed: bool, severity: Severity) -> GuaranteeEvaluation:
    return GuaranteeEvaluation(
        guarantee_id="g1",
        passed=passed,
        severity=severity,
        reason=None if passed else "failed",
        evaluator_id="test_ev",
        evaluator_version="1.0",
    )


def test_policy_accepts_all_passing():
    policy = StrictBehaviorPolicy()
    evals = [
        _make_evaluation(True, Severity.CRITICAL),
        _make_evaluation(True, Severity.HIGH),
    ]
    outcome, reasons = policy.decide(guarantees=evals)
    assert outcome == DecisionOutcome.ACCEPT
    assert reasons == ()


def test_policy_blocks_on_critical_failure():
    policy = StrictBehaviorPolicy()
    evals = [_make_evaluation(False, Severity.CRITICAL)]
    outcome, reasons = policy.decide(guarantees=evals)
    assert outcome == DecisionOutcome.BLOCK
    assert reasons[0].code == "critical_violation"


def test_policy_warns_on_two_high_failures():
    policy = StrictBehaviorPolicy()
    evals = [
        _make_evaluation(False, Severity.HIGH),
        _make_evaluation(False, Severity.HIGH),
    ]
    outcome, reasons = policy.decide(guarantees=evals)
    assert outcome == DecisionOutcome.WARN


def test_policy_accepts_on_one_high_failure():
    # One HIGH failure is not enough to warn — threshold is 2
    policy = StrictBehaviorPolicy()
    evals = [_make_evaluation(False, Severity.HIGH)]
    outcome, _ = policy.decide(guarantees=evals)
    assert outcome == DecisionOutcome.ACCEPT


def test_policy_critical_takes_precedence_over_high():
    policy = StrictBehaviorPolicy()
    evals = [
        _make_evaluation(False, Severity.CRITICAL),
        _make_evaluation(False, Severity.HIGH),
        _make_evaluation(False, Severity.HIGH),
    ]
    outcome, reasons = policy.decide(guarantees=evals)
    assert outcome == DecisionOutcome.BLOCK  # CRITICAL wins