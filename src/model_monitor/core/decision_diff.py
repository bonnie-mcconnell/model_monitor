"""Deterministic diff between two DecisionRecords for regression detection."""
from __future__ import annotations

from typing import Any

from model_monitor.contracts.behavioral.records import DecisionRecord


def diff_decisions(
    *,
    previous: DecisionRecord,
    current: DecisionRecord,
) -> dict[str, tuple[Any, Any]]:
    """
    Deterministically diff two decisions.
    Used for regression detection and audits.

    Returns a dict mapping field names to (previous_value, current_value) pairs.
    An empty dict means no change between the two records.
    """
    diffs: dict[str, tuple[Any, Any]] = {}

    if previous.outcome != current.outcome:
        diffs["outcome"] = (previous.outcome, current.outcome)

    if previous.reasons != current.reasons:
        diffs["reasons"] = (previous.reasons, current.reasons)

    return diffs
