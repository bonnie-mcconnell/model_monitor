"""
Tests for MetricsSummaryHistoryStore.

This is the append-only historical record of aggregated metrics - used
for trend charts and audits. The write path has never been tested.

The critical property: it must be append-only. Multiple writes for the
same window must each produce a new row, not overwrite the previous one.
That is what makes it a history, not a current-state store.
"""
from __future__ import annotations

import time

from model_monitor.storage.db import SessionLocal
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)
from model_monitor.storage.models.metrics_summary_history import (
    MetricsSummaryHistoryORM,
)


def _write_summary(
    store: MetricsSummaryHistoryStore,
    window: str = "5m",
    offset: float = 0.0,
) -> None:
    store.write(
        window=window,
        timestamp=time.time() + offset,
        n_batches=10,
        avg_accuracy=0.90,
        avg_f1=0.88,
        avg_confidence=0.85,
        avg_drift_score=0.03,
        avg_latency_ms=120.0,
    )


def test_write_persists_a_retrievable_row() -> None:
    """
    write() must commit a row that is visible via list_history().
    A write that silently discards the record would make the history
    endpoint always return empty without any visible error.
    """
    store = MetricsSummaryHistoryStore()
    unique_window = f"test_persist_{int(time.time() * 1000)}"
    _write_summary(store, window=unique_window)
    rows = store.list_history(window=unique_window)
    assert len(rows) == 1, (
        f"Expected 1 persisted row, got {len(rows)}"
    )


def test_write_is_append_only() -> None:
    """
    Writing twice for the same window must create two rows, not overwrite.
    This distinguishes MetricsSummaryHistoryStore (history) from
    MetricsSummaryStore (current state, upserts).
    """
    store = MetricsSummaryHistoryStore()
    ts = time.time() + 99999

    store.write(
        window="test_window_unique",
        timestamp=ts,
        n_batches=5,
        avg_accuracy=0.80,
        avg_f1=0.78,
        avg_confidence=0.75,
        avg_drift_score=0.05,
        avg_latency_ms=100.0,
    )
    store.write(
        window="test_window_unique",
        timestamp=ts + 1,
        n_batches=6,
        avg_accuracy=0.81,
        avg_f1=0.79,
        avg_confidence=0.76,
        avg_drift_score=0.04,
        avg_latency_ms=110.0,
    )

    session = SessionLocal()
    try:
        count = (
            session.query(MetricsSummaryHistoryORM)
            .filter(MetricsSummaryHistoryORM.window == "test_window_unique")
            .count()
        )
    finally:
        session.close()

    assert count == 2, (
        f"Expected 2 rows (append-only), got {count}. "
        "MetricsSummaryHistoryStore must not overwrite existing rows."
    )


def test_write_stores_correct_values() -> None:
    store = MetricsSummaryHistoryStore()
    ts = time.time() + 88888

    store.write(
        window="test_values",
        timestamp=ts,
        n_batches=42,
        avg_accuracy=0.91,
        avg_f1=0.89,
        avg_confidence=0.87,
        avg_drift_score=0.02,
        avg_latency_ms=95.0,
    )

    session = SessionLocal()
    try:
        row = (
            session.query(MetricsSummaryHistoryORM)
            .filter(
                MetricsSummaryHistoryORM.window == "test_values",
                MetricsSummaryHistoryORM.timestamp == ts,
            )
            .first()
        )
    finally:
        session.close()

    assert row is not None
    assert row.n_batches == 42
    assert abs(row.avg_accuracy - 0.91) < 1e-6
    assert abs(row.avg_f1 - 0.89) < 1e-6


# ---------------------------------------------------------------------------
# list_history - the public read path
# ---------------------------------------------------------------------------

def test_list_history_returns_oldest_first() -> None:
    """
    list_history must return rows in ascending timestamp order so callers
    receive a ready-to-plot time series without having to sort themselves.
    """
    store = MetricsSummaryHistoryStore()
    base_ts = time.time() + 77000

    for i in range(3):
        store.write(
            window="test_order",
            timestamp=base_ts + i,
            n_batches=i + 1,
            avg_accuracy=0.80,
            avg_f1=0.78,
            avg_confidence=0.75,
            avg_drift_score=0.05,
            avg_latency_ms=100.0,
        )

    rows = store.list_history(window="test_order", limit=10)

    assert len(rows) == 3
    timestamps = [r.timestamp for r in rows]
    assert timestamps == sorted(timestamps), (
        "list_history must return rows oldest-first; "
        f"got timestamps {timestamps}"
    )


def test_list_history_empty_window_returns_empty_list() -> None:
    """An unrecognised window name must return [] not raise."""
    store = MetricsSummaryHistoryStore()
    rows = store.list_history(window="__nonexistent_window__")
    assert rows == []


def test_list_history_limit_is_respected() -> None:
    """list_history(limit=2) must return at most 2 rows even when more exist."""
    store = MetricsSummaryHistoryStore()
    base_ts = time.time() + 66000

    for i in range(5):
        store.write(
            window="test_limit",
            timestamp=base_ts + i,
            n_batches=i + 1,
            avg_accuracy=0.85,
            avg_f1=0.83,
            avg_confidence=0.80,
            avg_drift_score=0.02,
            avg_latency_ms=90.0,
        )

    rows = store.list_history(window="test_limit", limit=2)

    assert len(rows) == 2, (
        f"limit=2 must cap results at 2 rows; got {len(rows)}"
    )


def test_list_history_rows_accessible_after_return() -> None:
    """
    ORM rows are expunged from the session before list_history returns.
    Accessing column values after the method returns must not raise
    DetachedInstanceError.
    """
    store = MetricsSummaryHistoryStore()
    ts = time.time() + 55000

    store.write(
        window="test_detach",
        timestamp=ts,
        n_batches=7,
        avg_accuracy=0.88,
        avg_f1=0.86,
        avg_confidence=0.83,
        avg_drift_score=0.01,
        avg_latency_ms=80.0,
    )

    rows = store.list_history(window="test_detach", limit=5)

    assert len(rows) >= 1
    # Access column values outside the session - must not raise
    row = rows[-1]
    assert row.n_batches == 7
    assert abs(row.avg_accuracy - 0.88) < 1e-6
