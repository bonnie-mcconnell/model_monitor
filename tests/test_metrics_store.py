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
from pathlib import Path
from typing import cast

from model_monitor.core.decisions import DecisionType
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
        "calibration_error": None,
        "feature_drift_scores": None,
        "behavioral_violation_rate": None,
        "shap_attribution": None,
        "action": DecisionType.NONE,
        "reason": "within thresholds",
        "previous_model": None,
        "new_model": None,
        # Fields added in later migrations - always supply them so the TypedDict
        # is fully satisfied even when running against the base schema.
        "p95_latency_ms": None,
        "p99_latency_ms": None,
        "output_drift_score": None,
        "output_drift_class_scores": None,
        "data_quality_score": None,
        "conformal_coverage": None,
        "conformal_set_size": None,
        "causal_drift_report": None,
        "mmd_p_value": None,
        "mmd_is_drift": None,
    }
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
    store.write(
        _make_record(timestamp=ts, action=DecisionType.RETRAIN, batch_id=retrain_id)
    )
    store.write(_make_record(timestamp=ts + 1, action=DecisionType.NONE))

    records, _ = store.list(limit=100, start_ts=ts - 1, action=DecisionType.RETRAIN)
    assert all(r["action"] == DecisionType.RETRAIN for r in records)
    assert any(r["batch_id"] == retrain_id for r in records)


def test_list_returns_none_cursor_when_no_more_results() -> None:
    store = MetricsStore()
    ts = time.time() + 30000
    store.write(_make_record(timestamp=ts))

    records, cursor = store.list(limit=100, start_ts=ts - 1)
    # Only one record in this window - must have no next page
    assert cursor is None


# ---------------------------------------------------------------------------
# behavioral_violation_rate round-trip
# ---------------------------------------------------------------------------


def test_behavioral_violation_rate_persists_and_roundtrips(tmp_path: Path) -> None:
    """
    behavioral_violation_rate written by MetricsStore.write() must be
    recovered unchanged by MetricsStore.tail().

    This is the persistence guarantee that allows the value to be surfaced
    in dashboards and Prometheus without re-running contract evaluations.
    """
    store = MetricsStore(db_path=tmp_path / "metrics.db")
    rate = 0.42
    store.write(_make_record(behavioral_violation_rate=rate))

    rows = store.tail(limit=1)
    assert len(rows) == 1
    assert rows[0]["behavioral_violation_rate"] == rate


def test_behavioral_violation_rate_none_when_not_configured(tmp_path: Path) -> None:
    """
    When no BehavioralContractRunner is configured, behavioral_violation_rate
    must be None in both the written and retrieved record.

    None is the correct sentinel for 'behavioral monitoring not active' -
    it is distinct from 0.0 ('active and no violations').
    """
    store = MetricsStore(db_path=tmp_path / "metrics.db")
    store.write(_make_record(behavioral_violation_rate=None))

    rows = store.tail(limit=1)
    assert len(rows) == 1
    assert rows[0]["behavioral_violation_rate"] is None


def test_behavioral_violation_rate_zero_distinct_from_none(tmp_path: Path) -> None:
    """
    0.0 (active monitoring, no violations) must roundtrip as 0.0, not None.
    The store must not collapse 0.0 to NULL.
    """
    store = MetricsStore(db_path=tmp_path / "metrics.db")
    store.write(_make_record(behavioral_violation_rate=0.0))

    rows = store.tail(limit=1)
    assert len(rows) == 1
    assert rows[0]["behavioral_violation_rate"] == 0.0
    assert rows[0]["behavioral_violation_rate"] is not None
