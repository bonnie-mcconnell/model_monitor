"""Guarantee and Severity - the building blocks of a contract."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """
    Severity of a guarantee violation.

    Determines the policy response via StrictBehaviorPolicy:
    CRITICAL → BLOCK immediately, HIGH → WARN after two failures, LOW → informational.
    """
    LOW = "low"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Guarantee:
    """
    A single enforceable guarantee within a behavioral contract.

    Binds a natural-language description to a specific evaluator (by ID)
    and declares the severity of a violation. Loaded from YAML; immutable
    at runtime.
    """
    guarantee_id: str
    description: str
    severity: Severity
    evaluator_id: str