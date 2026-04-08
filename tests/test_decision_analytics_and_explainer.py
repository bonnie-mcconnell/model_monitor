"""
Tests for DecisionAnalytics and DecisionExplainer.

These are presentation/analysis layers - they must not crash on empty
input, must produce correct summaries, and must map every decision
action to a human-readable rule name.
"""
from __future__ import annotations

import typing

from model_monitor.core.decision_analytics import DecisionAnalytics
from model_monitor.core.decision_explainer import DecisionExplainer
from model_monitor.core.decision_history import DecisionHistory
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decisions import Decision, DecisionMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decision(action: str, reason: str = "test") -> Decision:
    return Decision(action=action, reason=reason)  # type: ignore[arg-type]


def _make_snapshot(action: str) -> DecisionSnapshot:
    return DecisionSnapshot(
        decision_id="d1",
        action=action,  # type: ignore[arg-type]
        timestamp=0.0,
        status="executed",
    )


def _history_with(*actions: str) -> DecisionHistory:
    h = DecisionHistory()
    for a in actions:
        h.record(_make_decision(a))
    return h


# ---------------------------------------------------------------------------
# DecisionAnalytics
# ---------------------------------------------------------------------------

def test_analytics_summary_counts_actions() -> None:
    history = _history_with("none", "retrain", "none", "promote", "none")
    analytics = DecisionAnalytics(history)
    summary = analytics.decision_summary()
    assert summary["none"] == 3
    assert summary["retrain"] == 1
    assert summary["promote"] == 1


def test_analytics_summary_empty_history_returns_empty_dict() -> None:
    analytics = DecisionAnalytics(DecisionHistory())
    assert analytics.decision_summary() == {}


def test_analytics_tail_returns_last_n() -> None:
    history = _history_with("none", "retrain", "rollback")
    analytics = DecisionAnalytics(history)
    tail = analytics.decision_tail(limit=2)
    assert len(tail) == 2


def test_analytics_tail_empty_history_returns_empty() -> None:
    analytics = DecisionAnalytics(DecisionHistory())
    assert analytics.decision_tail() == []


def test_analytics_tail_returns_json_safe_dicts() -> None:
    """Keys must be strings - pandas returns Hashable by default."""
    history = _history_with("none")
    analytics = DecisionAnalytics(history)
    tail = analytics.decision_tail()
    for record in tail:
        for key in record:
            assert isinstance(key, str)


# ---------------------------------------------------------------------------
# DecisionExplainer
# ---------------------------------------------------------------------------

def test_explainer_maps_reject_to_severe_drift() -> None:
    explainer = DecisionExplainer()
    result = explainer.explain(
        decision=_make_decision("reject", "Severe drift"),
        snapshot=_make_snapshot("reject"),
    )
    assert result.rule_triggered == "severe_drift"


def test_explainer_maps_rollback_to_catastrophic_regression() -> None:
    explainer = DecisionExplainer()
    result = explainer.explain(
        decision=_make_decision("rollback", "F1 dropped"),
        snapshot=_make_snapshot("rollback"),
    )
    assert result.rule_triggered == "catastrophic_regression"


def test_explainer_maps_retrain_to_sustained_degradation() -> None:
    explainer = DecisionExplainer()
    result = explainer.explain(
        decision=_make_decision("retrain", "F1 drop"),
        snapshot=_make_snapshot("retrain"),
    )
    assert result.rule_triggered == "sustained_degradation"


def test_explainer_maps_promote_to_stability_promotion() -> None:
    explainer = DecisionExplainer()
    result = explainer.explain(
        decision=_make_decision("promote", "Stable"),
        snapshot=_make_snapshot("promote"),
    )
    assert result.rule_triggered == "stability_promotion"


def test_explainer_maps_none_to_within_thresholds() -> None:
    explainer = DecisionExplainer()
    result = explainer.explain(
        decision=_make_decision("none", "All good"),
        snapshot=_make_snapshot("none"),
    )
    assert result.rule_triggered == "within_thresholds"


def test_explainer_summary_equals_decision_reason() -> None:
    explainer = DecisionExplainer()
    reason = "Sustained performance degradation detected"
    result = explainer.explain(
        decision=_make_decision("retrain", reason),
        snapshot=_make_snapshot("retrain"),
    )
    assert result.summary == reason


def test_explainer_contributing_factors_contains_metadata() -> None:

    decision = Decision(
        action="retrain",
        reason="degraded",
        metadata=typing.cast(DecisionMetadata, {
            "trust_score": 0.55,
            "f1_drop": 0.08,
        }),
    )
    explainer = DecisionExplainer()
    result = explainer.explain(
        decision=decision,
        snapshot=_make_snapshot("retrain"),
    )
    assert result.contributing_factors["trust_score"] == 0.55
    assert result.contributing_factors["f1_drop"] == 0.08
