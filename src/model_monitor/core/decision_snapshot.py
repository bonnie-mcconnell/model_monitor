# src/model_monitor/core/decision_snapshot.py
from __future__ import annotations

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

from model_monitor.core.decisions import DecisionType


DecisionStatus = Literal["pending", "executed", "skipped", "failed"]


class DecisionSnapshot(BaseModel):
    """
    Persistent snapshot of a decision and its execution state.

    Used for:
    - idempotency
    - crash recovery
    - auditability
    """

    decision_id: str
    action: DecisionType
    timestamp: float = Field(..., gt=0)

    model_version: Optional[str] = None
    retrain_key: Optional[str] = None

    status: DecisionStatus = Field(
        ...,
        description="Execution status of the decision",
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True  # audit safety
