from dataclasses import dataclass
from typing import Tuple

from model_monitor.contracts.guarantee import Guarantee


@dataclass(frozen=True)
class Contract:
    contract_id: str
    version: str
    scope: str
    guarantees: Tuple[Guarantee, ...]