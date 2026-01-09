from dataclasses import dataclass
from typing import Literal, TypedDict


DecisionType = Literal["none", "retrain", "promote", "rollback", "reject"]


class DecisionMetadata(TypedDict, total=False):
    trust_score: float
    f1_drop: float
    baseline_f1: float
    current_f1: float
    drift_score: float
    threshold: float
    stable_batches: int
    cooldown_batches: int
    batches_since_last_retrain: int


@dataclass(frozen=True)
class Decision:
    """
    Represents a system-level operational decision.

    Lightweight, immutable, JSON-serializable.
    """

    action: DecisionType
    reason: str
    metadata: DecisionMetadata
