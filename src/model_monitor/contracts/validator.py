from .contract import Contract
from .registry import EvaluatorRegistry


def validate_contract(contract: Contract, registry: EvaluatorRegistry) -> None:
    for guarantee in contract.guarantees:
        registry.get(guarantee.evaluator_id)
