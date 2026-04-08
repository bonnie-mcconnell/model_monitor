"""Result types for individual evaluator runs."""
from __future__ import annotations

from dataclasses import dataclass

from model_monitor.contracts.guarantee import Severity


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Return value from a single GuaranteeEvaluator.evaluate() call."""
    passed: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class GuaranteeEvaluation:
    """
    Result of evaluating one guarantee, with full provenance.

    Produced by BehavioralContractRunner for each guarantee in the contract.
    Stored in DecisionRecord.guarantees - immutable audit evidence.
    """
    guarantee_id: str
    passed: bool
    severity: Severity
    reason: str | None
    evaluator_id: str
    evaluator_version: str