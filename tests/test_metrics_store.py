"""
Tests for MetricsStore - the primary persistence layer for batch metrics.

The cursor-based pagination in MetricsStore.list is the most complex
logic here: it uses a (timestamp, id) tuple to page through results
consistently under concurrent writes. This is non-trivial enough that
the test suite must cover it.
"""
from __future__ import annotations

import time
import uuid

from model_monitor.monitoring.types import MetricRecord
from model_monitor.storage.metrics_store import MetricsStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides: object) -> MetricRecord:
    base: MetricRecord = {
        "timestamp": time.time(),
        "batch_id": str(uuid.uuid4()),
        "n_samples": 100,
        "accuracy": 0.90,
        "f1": 0.88,
        "avg_confidence": 0.85,
        "drift_score": 0.03,
        "decision_latency_ms": 120.0,
        "action": "none",
        "reason": "within thresholds",
        "previous_model": None,
        "new_model": None,
    }
    from typing import cast
    return cast(MetricRecord, {**base, **overrides})


# ---------------------------------------------------------------------------
# write / tail / latest
# ---------------------------------------------------------------------------

def test_write_and_tail_returns_record() -> None:
    store = MetricsStore()
    record = _make_record()
    store.write(record)

    tail = store.tail(limit=1)
    assert len(tail) >= 1
    assert any(r["batch_id"] == record["batch_id"] for r in tail)


def test_latest_returns_most_recent() -> None:
    store = MetricsStore()
    early = _make_record(timestamp=time.time() - 100)
    late = _make_record(timestamp=time.time())
    store.write(early)
    store.write(late)

    latest = store.latest()
    assert latest is not None
    assert latest["batch_id"] == late["batch_id"]


def test_tail_empty_store_returns_empty_list() -> None:
    """
    We can't guarantee the store is empty (shared test DB), but we can
    verify the call does not raise and returns a list.
    """
    store = MetricsStore()
    result = store.tail(limit=0)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# list - cursor pagination
# ---------------------------------------------------------------------------

def test_list_without_cursor_returns_records() -> None:
    store = MetricsStore()
    record = _make_record()
    store.write(record)

    records, _ = store.list(limit=100)
    assert any(r["batch_id"] == record["batch_id"] for r in records)


def test_list_respects_limit() -> None:
    store = MetricsStore()
    ts = time.time()
    ids = []
    for i in range(5):
        r = _make_record(timestamp=ts + i)
        store.write(r)
        ids.append(r["batch_id"])

    records, next_cursor = store.list(limit=3, start_ts=ts - 1)
    assert len(records) <= 3


def test_list_cursor_paginates_consistently() -> None:
    """
    Cursor-based pagination must return every record exactly once.
    No records skipped, no records duplicated.
    """
    store = MetricsStore()
    ts = time.time() + 10000  # future ts to isolate from other test records
    written_ids = set()
    for i in range(6):
        r = _make_record(timestamp=ts + i)
        store.write(r)
        written_ids.add(r["batch_id"])

    collected_ids = set()
    cursor = None
    while True:
        records, cursor = store.list(limit=2, start_ts=ts - 1, cursor=cursor)
        for r in records:
            # No duplicates
            assert r["batch_id"] not in collected_ids
            collected_ids.add(r["batch_id"])
        if cursor is None:
            break

    assert written_ids.issubset(collected_ids)


def test_list_filter_by_action() -> None:
    store = MetricsStore()
    ts = time.time() + 20000
    retrain_id = str(uuid.uuid4())
    store.write(_make_record(timestamp=ts, action="retrain", batch_id=retrain_id))
    store.write(_make_record(timestamp=ts + 1, action="none"))

    records, _ = store.list(limit=100, start_ts=ts - 1, action="retrain")
    assert all(r["action"] == "retrain" for r in records)
    assert any(r["batch_id"] == retrain_id for r in records)


def test_list_returns_none_cursor_when_no_more_results() -> None:
    store = MetricsStore()
    ts = time.time() + 30000
    store.write(_make_record(timestamp=ts))

    records, cursor = store.list(limit=100, start_ts=ts - 1)
    # Only one record in this window - must have no next page
    assert cursor is None
