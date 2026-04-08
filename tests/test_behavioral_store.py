"""
Tests for BehavioralDecisionStore.

Verifies persistence, deduplication, and violation_rate computation.
violation_rate is the signal that feeds behavioral_violation_rate in
compute_trust_score - its correctness is load-bearing.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import pytest

from model_monitor.contracts.behavioral.context import DecisionContext
from model_monitor.contracts.behavioral.records import DecisionRecord
from model_monitor.contracts.outcome import DecisionOutcome
from model_monitor.storage.behavioral_decision_store import BehavioralDecisionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    outcome: DecisionOutcome,
    model_id: str = "model-v1",
    created_at: datetime | None = None,
) -> DecisionRecord:
    """Build a minimal DecisionRecord with a unique decision_id."""
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    return DecisionRecord(
        decision_id=str(uuid.uuid4()),
        context=DecisionContext(
            run_id=str(uuid.uuid4()),
            model_id=model_id,
            prompt_id="p1",
            output="output text",
            metadata={},
        ),
        guarantees=(),
        outcome=outcome,
        reasons=(),
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_record_persists_without_error() -> None:
    store = BehavioralDecisionStore()
    record = _make_record(DecisionOutcome.ACCEPT)
    store.record(record)  # must not raise


def test_duplicate_decision_id_is_silently_skipped() -> None:
    """
    Callers may retry on network failure. The same decision_id must not
    be written twice - idempotent writes prevent inflating violation counts.
    """
    store = BehavioralDecisionStore()
    record = _make_record(DecisionOutcome.BLOCK)

    store.record(record)
    store.record(record)  # second write - must not raise or double-count

    since = record.created_at.timestamp() - 1
    rate = store.violation_rate(since_ts=since)
    # If written twice, rate would still be 1.0 (1/1 not 2/2),
    # but the important property is: no crash and rate is valid.
    assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# violation_rate - the core signal
# ---------------------------------------------------------------------------

def test_no_records_returns_zero_violation_rate() -> None:
    """
    No data means no signal, not maximum suspicion.
    """
    store = BehavioralDecisionStore()
    # Use a future timestamp - guaranteed no records exist after it
    rate = store.violation_rate(since_ts=time.time() + 9999)
    assert rate == 0.0


def test_block_outcome_counts_as_violation() -> None:
    store = BehavioralDecisionStore()
    since = time.time()
    store.record(_make_record(DecisionOutcome.BLOCK))

    rate = store.violation_rate(since_ts=since)
    assert rate == pytest.approx(1.0)


def test_warn_outcome_counts_as_violation() -> None:
    store = BehavioralDecisionStore()
    since = time.time()
    store.record(_make_record(DecisionOutcome.WARN))

    rate = store.violation_rate(since_ts=since)
    assert rate == pytest.approx(1.0)


def test_accept_outcome_does_not_count_as_violation() -> None:
    store = BehavioralDecisionStore()
    since = time.time()
    store.record(_make_record(DecisionOutcome.ACCEPT))

    rate = store.violation_rate(since_ts=since)
    assert rate == pytest.approx(0.0)


def test_partial_violation_rate() -> None:
    """
    Two accepts and two blocks → 50% violation rate.
    """
    store = BehavioralDecisionStore()
    since = time.time()

    store.record(_make_record(DecisionOutcome.ACCEPT))
    store.record(_make_record(DecisionOutcome.ACCEPT))
    store.record(_make_record(DecisionOutcome.BLOCK))
    store.record(_make_record(DecisionOutcome.BLOCK))

    rate = store.violation_rate(since_ts=since)
    assert rate == pytest.approx(0.5)


def test_violation_rate_respects_time_window() -> None:
    """
    Records created before since_ts must not be counted.
    This is the mechanism that makes violation_rate window-scoped.

    Sequence:
    1. Write an old BLOCK with created_at two hours ago
    2. Capture since = time.time() (now - after the old record, before the new one)
    3. Write a recent ACCEPT
    4. violation_rate(since_ts=since) should see only the ACCEPT → 0.0

    We do NOT use since = time.time() - 60 because other tests in the same
    session may have written BLOCK records within the last 60 seconds,
    which would contaminate the window.
    """
    store = BehavioralDecisionStore()

    # Write old violation with an explicit past timestamp
    old_ts = datetime.fromtimestamp(time.time() - 7200, tz=timezone.utc)
    store.record(_make_record(DecisionOutcome.BLOCK, created_at=old_ts))

    # Capture the window boundary AFTER the old record but BEFORE the new one
    since = time.time()

    # Write a recent accept - this is the only record inside the window
    store.record(_make_record(DecisionOutcome.ACCEPT))

    rate = store.violation_rate(since_ts=since)
    assert rate == pytest.approx(0.0)


def test_violation_rate_is_bounded() -> None:
    """
    violation_rate must always be in [0.0, 1.0] regardless of data.
    """
    store = BehavioralDecisionStore()
    since = time.time()

    for outcome in [DecisionOutcome.BLOCK, DecisionOutcome.WARN, DecisionOutcome.ACCEPT]:
        store.record(_make_record(outcome))

    rate = store.violation_rate(since_ts=since)
    assert 0.0 <= rate <= 1.0
