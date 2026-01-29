from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    LOW = "low"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Guarantee:
    guarantee_id: str
    description: str
    severity: Severity
    evaluator_id: str
