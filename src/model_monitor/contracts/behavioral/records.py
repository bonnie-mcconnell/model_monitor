"""Immutable audit record produced by BehavioralContractRunner."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from model_monitor.contracts.behavioral.context import DecisionContext
from model_monitor.contracts.behavioral.evaluation import GuaranteeEvaluation
from model_monitor.contracts.outcome import DecisionOutcome, OutcomeReason


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """
    Immutable audit record for a single behavioral contract evaluation.

    Created by BehavioralContractRunner.evaluate(). Contains full provenance:
    which evaluators ran, their versions, what each returned, the policy
    outcome, and a UTC timestamp. decision_id is a UUID used for deduplication
    in BehavioralDecisionStore.
    """
    decision_id: str
    context: DecisionContext
    guarantees: tuple[GuaranteeEvaluation, ...]
    outcome: DecisionOutcome
    reasons: tuple[OutcomeReason, ...]
    created_at: datetime