"""
Tests for decision_diff and DecisionHistory.

decision_diff is the behavioral regression detection function - it tells
you exactly what changed between two consecutive evaluations. If it
silently returns empty diffs when things changed, regressions go undetected.

DecisionHistory is the in-memory rolling buffer that feeds the decision
engine's hysteresis logic. Its maxlen bound matters: overflow must evict
oldest entries, not crash.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from model_monitor.contracts.behavioral.context import DecisionContext
from model_monitor.contracts.behavioral.records import DecisionRecord
from model_monitor.contracts.outcome import DecisionOutcome, OutcomeReason
from model_monitor.core.decision_diff import diff_decisions
from model_monitor.core.decision_history import DecisionHistory
from model_monitor.core.decisions import Decision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_behavioral_record(
    outcome: DecisionOutcome,
    reasons: tuple[OutcomeReason, ...] = (),
) -> DecisionRecord:
    return DecisionRecord(
        decision_id=str(uuid.uuid4()),
        context=DecisionContext(
            run_id=str(uuid.uuid4()),
            model_id="m1",
            prompt_id="p1",
            output="output",
            metadata={},
        ),
        guarantees=(),
        outcome=outcome,
        reasons=reasons,
        created_at=datetime.now(timezone.utc),
    )


def _make_decision(action: str = "none") -> Decision:
    return Decision(action=action, reason="test reason")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# diff_decisions
# ---------------------------------------------------------------------------

def test_diff_returns_empty_when_outcomes_identical() -> None:
    prev = _make_behavioral_record(DecisionOutcome.ACCEPT)
    curr = _make_behavioral_record(DecisionOutcome.ACCEPT)
    assert diff_decisions(previous=prev, current=curr) == {}


def test_diff_detects_outcome_change() -> None:
    prev = _make_behavioral_record(DecisionOutcome.ACCEPT)
    curr = _make_behavioral_record(DecisionOutcome.BLOCK)

    diffs = diff_decisions(previous=prev, current=curr)

    assert "outcome" in diffs
    assert diffs["outcome"] == (DecisionOutcome.ACCEPT, DecisionOutcome.BLOCK)


def test_diff_detects_reasons_change() -> None:
    r1 = OutcomeReason(code="c1", message="m1")
    r2 = OutcomeReason(code="c2", message="m2")

    prev = _make_behavioral_record(DecisionOutcome.BLOCK, reasons=(r1,))
    curr = _make_behavioral_record(DecisionOutcome.BLOCK, reasons=(r2,))

    diffs = diff_decisions(previous=prev, current=curr)

    assert "reasons" in diffs
    assert diffs["reasons"][0] == (r1,)
    assert diffs["reasons"][1] == (r2,)


def test_diff_detects_both_outcome_and_reasons_change() -> None:
    r2 = OutcomeReason(code="c2", message="after")

    prev = _make_behavioral_record(DecisionOutcome.ACCEPT)
    curr = _make_behavioral_record(DecisionOutcome.WARN, reasons=(r2,))

    diffs = diff_decisions(previous=prev, current=curr)

    assert "outcome" in diffs
    assert "reasons" in diffs


def test_diff_is_deterministic() -> None:
    """Same inputs must always produce the same output."""
    prev = _make_behavioral_record(DecisionOutcome.ACCEPT)
    curr = _make_behavioral_record(DecisionOutcome.BLOCK)

    assert (
        diff_decisions(previous=prev, current=curr)
        == diff_decisions(previous=prev, current=curr)
    )


# ---------------------------------------------------------------------------
# DecisionHistory
# ---------------------------------------------------------------------------

def test_history_records_and_retrieves_actions() -> None:
    history = DecisionHistory()
    history.record(_make_decision("retrain"))
    history.record(_make_decision("none"))

    actions = history.recent_actions()
    assert actions == ["retrain", "none"]


def test_history_respects_maxlen_bound() -> None:
    """
    DecisionHistory uses deque(maxlen=N). Adding more than N decisions
    must evict the oldest, not crash or silently grow without bound.
    """
    history = DecisionHistory(maxlen=3)
    for action in ["retrain", "none", "none", "promote"]:
        history.record(_make_decision(action))

    actions = history.recent_actions()
    # Oldest ("retrain") evicted, last 3 remain
    assert len(actions) == 3
    assert "retrain" not in actions
    assert actions == ["none", "none", "promote"]


def test_history_recent_actions_with_limit() -> None:
    history = DecisionHistory()
    for action in ["retrain", "none", "none", "promote"]:
        history.record(_make_decision(action))

    last_two = history.recent_actions(limit=2)
    assert last_two == ["none", "promote"]


def test_history_tail_returns_decision_objects() -> None:
    history = DecisionHistory()
    d = _make_decision("rollback")
    history.record(d)

    tail = history.tail(limit=1)
    assert len(tail) == 1
    assert tail[0].action == "rollback"


def test_history_empty_returns_empty_list() -> None:
    history = DecisionHistory()
    assert history.recent_actions() == []
    assert history.tail() == []


def test_history_is_iterable() -> None:
    history = DecisionHistory()
    history.record(_make_decision("none"))
    history.record(_make_decision("promote"))

    actions = [d.action for d in history]
    assert actions == ["none", "promote"]
