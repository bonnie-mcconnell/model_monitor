import yaml

from model_monitor.contracts.contract import Contract
from model_monitor.contracts.guarantee import Guarantee, Severity


def load_contract_from_yaml(path: str) -> Contract:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    guarantees = tuple(
        Guarantee(
            guarantee_id=g["id"],
            description=g["description"],
            severity=Severity(g["severity"].lower()),
            evaluator_id=g["evaluator"],
        )
        for g in raw["guarantees"]
    )

    return Contract(
        contract_id=raw["contract_id"],
        version=str(raw["version"]),
        scope=raw["scope"],
        guarantees=guarantees,
    )