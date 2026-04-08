"""UI boundary adapter: construct and format Decision objects for display."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_monitor.core.decisions import Decision, DecisionMetadata


def _normalize_payload(payload: Mapping[Any, Any]) -> dict[str, Any]:
    """
    Normalize API / pandas payloads to string-keyed dicts.

    Pandas returns dict[Hashable, Any] which is correct but
    too weak for domain construction.
    """
    return {str(k): v for k, v in payload.items()}


def decision_from_api(payload: Mapping[Any, Any]) -> Decision:
    """
    Safely construct a Decision from an API payload.
    UI boundary adapter.
    """
    data = _normalize_payload(payload)

    return Decision(
        action=data["action"],
        reason=data["reason"],
        metadata=data.get("metadata", {}),
    )


def format_decision_explanation(decision: Decision) -> dict[str, str]:
    """
    Convert a Decision into a human-readable explanation
    suitable for UI, logs, and demos.
    """
    action = decision.action.replace("_", " ").title()
    reason = decision.reason
    md: DecisionMetadata = decision.metadata or {}

    bullets: list[str] = []

    if "trust_score" in md:
        bullets.append(f"Trust score: {md['trust_score']:.3f}")

    if "f1_drop" in md:
        bullets.append(f"F1 drop: {md['f1_drop']:.3f}")

    if "baseline_f1" in md and "current_f1" in md:
        bullets.append(
            f"F1: {md['current_f1']:.3f} → {md['baseline_f1']:.3f}"
        )

    if "drift_score" in md:
        bullets.append(f"Drift score: {md['drift_score']:.3f}")

    if "cooldown_batches" in md:
        bullets.append(
            f"Cooldown remaining: {md['cooldown_batches']} batches"
        )

    if not bullets:
        bullets.append("No diagnostic metadata")

    return {
        "title": action,
        "reason": reason,
        "details": " • ".join(bullets),
    }

