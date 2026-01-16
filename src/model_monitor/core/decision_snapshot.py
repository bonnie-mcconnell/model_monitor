# src/model_monitor/core/decision_snapshot.py
from __future__ import annotations

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from model_monitor.core.decisions import DecisionType


class DecisionSnapshot(BaseModel):
    """
    Persistent snapshot of a decision and its execution state.

    This object is intentionally MUTABLE.

    It represents the lifecycle of a decision execution and is used for:
    - idempotency
    - crash recovery
    - auditing
    """

    model_config = ConfigDict(
        frozen=False,          # <-- REQUIRED (executor mutates state)
        validate_assignment=True,
        extra="forbid",
    )

    decision_id: str = Field(..., description="Globally unique decision ID")
    action: DecisionType = Field(..., description="Decision action taken")
    timestamp: float = Field(..., description="Decision creation time (epoch)")

    # Execution state
    status: str = Field(
        ...,
        description="Execution status: pending | executed | skipped | failed",
    )

    # Optional execution context
    model_version: Optional[str] = Field(
        default=None,
        description="Model version active at execution time",
    )

    retrain_key: Optional[str] = Field(
        default=None,
        description="Unique retrain artifact key (if retrain triggered)",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form execution metadata",
    )
