from __future__ import annotations

from typing import Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field

from model_monitor.core.decisions import DecisionType

@dataclass
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
    timestamp: float

    model_version: Optional[str] = None

    retrain_key: Optional[str] = None
    status: str = Field(
        ...,
        description="Execution status: pending | executed | failed",
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)
