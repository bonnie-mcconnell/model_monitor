"""
Tests for the behavioral contract infrastructure:
EvaluatorRegistry, load_contract_from_yaml, validate_contract.

These are the load-bearing plumbing tests. They are not exciting but
they matter: a misregistered evaluator or a badly-formed YAML silently
passes at load time and explodes mid-evaluation under production load.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from model_monitor.contracts.behavioral.evaluation import EvaluationResult
from model_monitor.contracts.behavioral.evaluators import JsonValidityEvaluator
from model_monitor.contracts.guarantee import Severity
from model_monitor.contracts.loader import load_contract_from_yaml
from model_monitor.contracts.registry import EvaluatorRegistry
from model_monitor.contracts.validator import validate_contract

# ---------------------------------------------------------------------------
# EvaluatorRegistry
# ---------------------------------------------------------------------------

class _StubEvaluator:
    """Minimal evaluator that satisfies the GuaranteeEvaluator Protocol."""
    def __init__(self, evaluator_id: str) -> None:
        self.evaluator_id = evaluator_id
        self.version = "1.0"

    def evaluate(self, *, output: str) -> EvaluationResult:
        return EvaluationResult(passed=True)


def test_registry_register_and_retrieve() -> None:
    registry = EvaluatorRegistry()
    ev = _StubEvaluator("stub_v1")
    registry.register(ev)
    assert registry.get("stub_v1") is ev


def test_registry_raises_on_duplicate_id() -> None:
    registry = EvaluatorRegistry()
    registry.register(_StubEvaluator("dup"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_StubEvaluator("dup"))


def test_registry_raises_on_unknown_id() -> None:
    registry = EvaluatorRegistry()
    with pytest.raises(KeyError, match="Unknown evaluator"):
        registry.get("does_not_exist")


def test_registry_different_ids_do_not_conflict() -> None:
    registry = EvaluatorRegistry()
    ev_a = _StubEvaluator("ev_a")
    ev_b = _StubEvaluator("ev_b")
    registry.register(ev_a)
    registry.register(ev_b)
    assert registry.get("ev_a") is ev_a
    assert registry.get("ev_b") is ev_b


def test_registry_is_append_only() -> None:
    """
    Registering a new evaluator with an existing ID must fail even if the
    new evaluator has a different version. Append-only means IDs are permanent.
    """
    registry = EvaluatorRegistry()
    registry.register(_StubEvaluator("ev_v1"))

    v2 = _StubEvaluator("ev_v1")
    v2.version = "2.0"
    with pytest.raises(ValueError):
        registry.register(v2)


# ---------------------------------------------------------------------------
# load_contract_from_yaml
# ---------------------------------------------------------------------------

def _write_contract_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "contract.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def test_load_contract_from_yaml_parses_correctly(tmp_path: Path) -> None:
    path = _write_contract_yaml(tmp_path, """
        contract_id: test_contract_v1
        version: "1.0"
        scope: test_scope
        guarantees:
          - id: check_json
            description: Must be valid JSON
            severity: critical
            evaluator: json_validity
          - id: check_schema
            description: Must match schema
            severity: high
            evaluator: json_schema_v1
    """)

    contract = load_contract_from_yaml(str(path))

    assert contract.contract_id == "test_contract_v1"
    assert contract.version == "1.0"
    assert contract.scope == "test_scope"
    assert len(contract.guarantees) == 2


def test_load_contract_severity_is_enum(tmp_path: Path) -> None:
    path = _write_contract_yaml(tmp_path, """
        contract_id: c1
        version: "1.0"
        scope: s1
        guarantees:
          - id: g1
            description: d1
            severity: CRITICAL
            evaluator: ev1
    """)
    contract = load_contract_from_yaml(str(path))
    assert contract.guarantees[0].severity == Severity.CRITICAL


def test_load_contract_severity_case_insensitive(tmp_path: Path) -> None:
    """The loader normalises severity to lowercase before constructing the Enum."""
    for raw_severity, expected in [("critical", Severity.CRITICAL),
                                    ("HIGH", Severity.HIGH),
                                    ("Low", Severity.LOW)]:
        path = _write_contract_yaml(tmp_path, f"""
            contract_id: c1
            version: "1.0"
            scope: s1
            guarantees:
              - id: g1
                description: d
                severity: {raw_severity}
                evaluator: ev1
        """)
        contract = load_contract_from_yaml(str(path))
        assert contract.guarantees[0].severity == expected


def test_load_contract_version_coerced_to_string(tmp_path: Path) -> None:
    """
    YAML parses 'version: 1.0' as a float. The loader must coerce it to
    a string - otherwise contract_id comparisons and logging break silently.
    """
    path = _write_contract_yaml(tmp_path, """
        contract_id: c1
        version: 1.0
        scope: s1
        guarantees:
          - id: g1
            description: d
            severity: low
            evaluator: ev1
    """)
    contract = load_contract_from_yaml(str(path))
    assert isinstance(contract.version, str)
    assert contract.version == "1.0"


def test_load_contract_guarantees_are_immutable_tuple(tmp_path: Path) -> None:
    """Guarantees must be a tuple, not a list - contracts must not be mutated."""
    path = _write_contract_yaml(tmp_path, """
        contract_id: c1
        version: "1.0"
        scope: s1
        guarantees:
          - id: g1
            description: d
            severity: low
            evaluator: ev1
    """)
    contract = load_contract_from_yaml(str(path))
    assert isinstance(contract.guarantees, tuple)


# ---------------------------------------------------------------------------
# validate_contract
# ---------------------------------------------------------------------------

def test_validate_contract_passes_when_all_evaluators_registered(
    tmp_path: Path,
) -> None:
    path = _write_contract_yaml(tmp_path, """
        contract_id: c1
        version: "1.0"
        scope: s1
        guarantees:
          - id: g1
            description: d
            severity: critical
            evaluator: json_validity
    """)
    contract = load_contract_from_yaml(str(path))
    registry = EvaluatorRegistry()
    registry.register(JsonValidityEvaluator())

    # Must not raise
    validate_contract(contract, registry)


def test_validate_contract_raises_when_evaluator_missing(
    tmp_path: Path,
) -> None:
    """
    validate_contract is the pre-flight check. It must raise immediately
    when a contract references an evaluator that is not registered, so the
    failure surfaces at startup rather than mid-evaluation.
    """
    path = _write_contract_yaml(tmp_path, """
        contract_id: c1
        version: "1.0"
        scope: s1
        guarantees:
          - id: g1
            description: d
            severity: critical
            evaluator: not_registered
    """)
    contract = load_contract_from_yaml(str(path))
    registry = EvaluatorRegistry()

    with pytest.raises(KeyError, match="not_registered"):
        validate_contract(contract, registry)
