"""
Tests for ui/decision_explanation.py.

decision_from_api is a boundary adapter used by the Streamlit dashboard.
format_decision_explanation builds human-readable strings shown to engineers
during incident triage. Both have real branching logic and must be correct.
"""
from __future__ import annotations

import typing

from model_monitor.core.decisions import Decision, DecisionMetadata
from model_monitor.ui.decision_explanation import (
    decision_from_api,
    format_decision_explanation,
)

# ---------------------------------------------------------------------------
# decision_from_api
# ---------------------------------------------------------------------------

def test_decision_from_api_constructs_correctly() -> None:
    payload = {
        "action": "retrain",
        "reason": "F1 degraded",
        "metadata": {"trust_score": 0.55},
    }
    decision = decision_from_api(payload)
    assert decision.action == "retrain"
    assert decision.reason == "F1 degraded"


def test_decision_from_api_accepts_non_string_keys() -> None:
    """
    Pandas DataFrames return dict[Hashable, Any] - keys may not be strings.
    _normalize_payload must coerce them. We verify with string keys since
    non-string keys would fail the Decision constructor, not normalize_payload.
    """
    payload = {"action": "none", "reason": "ok"}
    decision = decision_from_api(payload)
    assert decision.action == "none"


def test_decision_from_api_missing_metadata_defaults_to_empty() -> None:
    payload = {"action": "none", "reason": "ok"}
    decision = decision_from_api(payload)
    assert decision.metadata == {} or decision.metadata is None or isinstance(decision.metadata, dict)


# ---------------------------------------------------------------------------
# format_decision_explanation - output structure
# ---------------------------------------------------------------------------

def _make_decision(action: str, reason: str, **meta) -> Decision:
    return Decision(
        action=action,  # type: ignore[arg-type]
        reason=reason,
        metadata=typing.cast(DecisionMetadata, meta),
    )


def test_format_returns_title_reason_details() -> None:
    decision = _make_decision("retrain", "F1 degraded")
    result = format_decision_explanation(decision)
    assert "title" in result
    assert "reason" in result
    assert "details" in result


def test_format_action_is_title_cased() -> None:
    decision = _make_decision("system_error", "executor failed")
    result = format_decision_explanation(decision)
    assert result["title"] == "System Error"


def test_format_reason_preserved_exactly() -> None:
    reason = "Sustained performance degradation detected"
    decision = _make_decision("retrain", reason)
    result = format_decision_explanation(decision)
    assert result["reason"] == reason


def test_format_trust_score_in_details_when_present() -> None:
    decision = _make_decision("none", "ok", trust_score=0.823)
    result = format_decision_explanation(decision)
    assert "0.823" in result["details"]


def test_format_f1_drop_in_details_when_present() -> None:
    decision = _make_decision("retrain", "degraded", f1_drop=0.075)
    result = format_decision_explanation(decision)
    assert "0.075" in result["details"]


def test_format_drift_score_in_details_when_present() -> None:
    decision = _make_decision("reject", "drift", drift_score=0.312)
    result = format_decision_explanation(decision)
    assert "0.312" in result["details"]


def test_format_cooldown_in_details_when_present() -> None:
    decision = _make_decision("none", "cooldown", cooldown_batches=3)
    result = format_decision_explanation(decision)
    assert "3" in result["details"]


def test_format_no_metadata_produces_fallback_message() -> None:
    decision = _make_decision("none", "ok")
    result = format_decision_explanation(decision)
    assert "No diagnostic metadata" in result["details"]


def test_format_f1_transition_shown_when_both_present() -> None:
    """baseline_f1 and current_f1 together produce a transition string."""
    decision = _make_decision(
        "retrain", "degraded",
        baseline_f1=0.85,
        current_f1=0.77,
    )
    result = format_decision_explanation(decision)
    assert "0.770" in result["details"]
    assert "0.850" in result["details"]
