from dataclasses import dataclass
from datetime import datetime

from .context import DecisionContext
from .evaluation import GuaranteeEvaluation
from ..outcome import DecisionOutcome, OutcomeReason


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    decision_id: str
    context: DecisionContext
    guarantees: tuple[GuaranteeEvaluation, ...]
    outcome: DecisionOutcome
    reasons: tuple[OutcomeReason, ...]
    created_at: datetime
