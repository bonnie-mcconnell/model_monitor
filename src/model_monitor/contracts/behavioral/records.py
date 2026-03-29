from dataclasses import dataclass
from datetime import datetime

from model_monitor.contracts.behavioral.context import DecisionContext
from model_monitor.contracts.behavioral.evaluation import GuaranteeEvaluation
from model_monitor.contracts.outcome import DecisionOutcome, OutcomeReason


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    decision_id: str
    context: DecisionContext
    guarantees: tuple[GuaranteeEvaluation, ...]
    outcome: DecisionOutcome
    reasons: tuple[OutcomeReason, ...]
    created_at: datetime