"""Pre-flight validation that all contract evaluators are registered."""
from __future__ import annotations

from model_monitor.contracts.contract import Contract
from model_monitor.contracts.registry import EvaluatorRegistry


def validate_contract(contract: Contract, registry: EvaluatorRegistry) -> None:
    for guarantee in contract.guarantees:
        registry.get(guarantee.evaluator_id)