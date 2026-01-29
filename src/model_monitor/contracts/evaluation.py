from dataclasses import dataclass
from typing import Optional

from .guarantee import Severity


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    passed: bool
    reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class GuaranteeEvaluation:
    guarantee_id: str
    passed: bool
    severity: Severity
    reason: Optional[str]
    evaluator_id: str
    evaluator_version: str
