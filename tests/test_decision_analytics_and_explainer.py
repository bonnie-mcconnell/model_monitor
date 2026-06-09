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
from model_monitor.core.decisions import Decision, DecisionMetadata, DecisionType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_decision(action: DecisionType, reason: str = "test") -> Decision:
    return Decision(action=action, reason=reason)


def _make_snapshot(action: DecisionType) -> DecisionSnapshot:
    return DecisionSnapshot(
        decision_id="d1",
        action=action,
        timestamp=0.0,
        status="executed",
    )


def _history_with(*actions: DecisionType) -> DecisionHistory:
    h = DecisionHistory()
    for a in actions:
        h.record(_make_decision(a))
    return h


# ---------------------------------------------------------------------------
# DecisionAnalytics
# ---------------------------------------------------------------------------


def test_analytics_summary_counts_actions() -> None:
    history = _history_with(
        DecisionType.NONE,
        DecisionType.RETRAIN,
        DecisionType.NONE,
        DecisionType.PROMOTE,
        DecisionType.NONE,
    )
    analytics = DecisionAnalytics(history)
    summary = analytics.decision_summary()
    assert summary[DecisionType.NONE] == 3
    assert summary[DecisionType.RETRAIN] == 1
    assert summary[DecisionType.PROMOTE] == 1


def test_analytics_summary_empty_history_returns_empty_dict() -> None:
    analytics = DecisionAnalytics(DecisionHistory())
    assert analytics.decision_summary() == {}


def test_analytics_tail_returns_last_n() -> None:
    history = _history_with(
        DecisionType.NONE, DecisionType.RETRAIN, DecisionType.ROLLBACK
    )
    analytics = DecisionAnalytics(history)
    tail = analytics.decision_tail(limit=2)
    assert len(tail) == 2


def test_analytics_tail_empty_history_returns_empty() -> None:
    analytics = DecisionAnalytics(DecisionHistory())
    assert analytics.decision_tail() == []


def test_analytics_tail_returns_json_safe_dicts() -> None:
    """Keys must be strings - pandas returns Hashable by default."""
    history = _history_with(DecisionType.NONE)
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
        decision=_make_decision(DecisionType.REJECT, "Severe drift"),
        snapshot=_make_snapshot(DecisionType.REJECT),
    )
    assert result.rule_triggered == "severe_drift"


def test_explainer_maps_rollback_to_catastrophic_regression() -> None:
    explainer = DecisionExplainer()
    result = explainer.explain(
        decision=_make_decision(DecisionType.ROLLBACK, "F1 dropped"),
        snapshot=_make_snapshot(DecisionType.ROLLBACK),
    )
    assert result.rule_triggered == "catastrophic_regression"


def test_explainer_maps_retrain_to_sustained_degradation() -> None:
    explainer = DecisionExplainer()
    result = explainer.explain(
        decision=_make_decision(DecisionType.RETRAIN, "F1 drop"),
        snapshot=_make_snapshot(DecisionType.RETRAIN),
    )
    assert result.rule_triggered == "sustained_degradation"


def test_explainer_maps_promote_to_stability_promotion() -> None:
    explainer = DecisionExplainer()
    result = explainer.explain(
        decision=_make_decision(DecisionType.PROMOTE, "Stable"),
        snapshot=_make_snapshot(DecisionType.PROMOTE),
    )
    assert result.rule_triggered == "stability_promotion"


def test_explainer_maps_none_to_within_thresholds() -> None:
    explainer = DecisionExplainer()
    result = explainer.explain(
        decision=_make_decision(DecisionType.NONE, "All good"),
        snapshot=_make_snapshot(DecisionType.NONE),
    )
    assert result.rule_triggered == "within_thresholds"


def test_explainer_summary_equals_decision_reason() -> None:
    explainer = DecisionExplainer()
    reason = "Sustained performance degradation detected"
    result = explainer.explain(
        decision=_make_decision(DecisionType.RETRAIN, reason),
        snapshot=_make_snapshot(DecisionType.RETRAIN),
    )
    assert result.summary == reason


def test_explainer_contributing_factors_contains_metadata() -> None:

    decision = Decision(
        action=DecisionType.RETRAIN,
        reason="degraded",
        metadata=typing.cast(
            DecisionMetadata,
            {
                "trust_score": 0.55,
                "f1_drop": 0.08,
            },
        ),
    )
    explainer = DecisionExplainer()
    result = explainer.explain(
        decision=decision,
        snapshot=_make_snapshot(DecisionType.RETRAIN),
    )
    assert result.contributing_factors["trust_score"] == 0.55
    assert result.contributing_factors["f1_drop"] == 0.08
