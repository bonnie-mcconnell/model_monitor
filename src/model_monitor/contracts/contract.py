"""Contract dataclass - the versioned behavioral specification."""
from __future__ import annotations

from dataclasses import dataclass

from model_monitor.contracts.guarantee import Guarantee


@dataclass(frozen=True)
class Contract:
    """
    A versioned behavioral contract for a model.

    Loaded once at startup from a YAML file; never mutated at runtime.
    A contract is the unit of registration: one contract_id maps to one
    set of guarantees enforced by BehavioralContractRunner.
    """
    contract_id: str
    version: str
    scope: str
    guarantees: tuple[Guarantee, ...]