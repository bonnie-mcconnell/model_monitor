from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from model_monitor.core.decisions import DecisionType


@dataclass(frozen=True)
class DecisionSnapshot:
    """
    Immutable snapshot of system state at decision time.
    """

    batch_index: int
    trust_score: float
    f1: float
    f1_baseline: float
    drift_score: float
    recent_actions: Sequence[DecisionType] | None
    captured_at: float
