from dataclasses import dataclass
from enum import Enum


class DecisionOutcome(Enum):
    ACCEPT = "accept"
    WARN = "warn"
    BLOCK = "block"
    ROLLBACK = "rollback"


@dataclass(frozen=True, slots=True)
class OutcomeReason:
    code: str
    message: str