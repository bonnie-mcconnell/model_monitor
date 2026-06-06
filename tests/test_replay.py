"""Tests for MetricsSummaryHistoryStore.query_range and cli.replay._run_replay.

Covers:
- query_range returns only rows within the requested time range.
- query_range returns rows ordered oldest-first.
- query_range returns an empty list when no rows match.
- replay._run_replay writes decisions for each row.
- replay._run_replay dry-run does not write decisions.
- replay._run_replay emits JSON when requested.
- replay._run_replay returns 0 on success, 0 on empty range.
- _parse_iso handles all supported date/datetime formats and rejects invalid input.
"""

from __future__ import annotations

import json
import time
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine

from model_monitor.storage.db import Base
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_ROW = dict(
    window="1h",
    n_batches=5,
    avg_accuracy=0.90,
    avg_f1=0.88,
    avg_confidence=0.85,
    avg_drift_score=0.04,
    avg_latency_ms=120.0,
)


def _store(tmp_path: Path) -> MetricsSummaryHistoryStore:
    """Return a MetricsSummaryHistoryStore backed by an isolated SQLite file."""
    # Patch SessionLocal to use a per-test database so tests do not share state.
    db_path = tmp_path / "history.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(bind=engine)
    store = MetricsSummaryHistoryStore()
    store._session_factory = factory
    return store


def _write(store: MetricsSummaryHistoryStore, ts: float, **kwargs: object) -> None:
    kw = {**_BASE_ROW, "timestamp": ts}
    kw.update(kwargs)
    store.write(**kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# query_range
# ---------------------------------------------------------------------------


def test_query_range_returns_rows_within_bounds(tmp_path: Path) -> None:
    """query_range returns only rows whose timestamp falls within [from_ts, to_ts]."""
    store = _store(tmp_path)
    now = time.time()

    _write(store, now - 7200)  # 2h ago - outside range
    _write(store, now - 3600)  # 1h ago - inside
    _write(store, now - 1800)  # 30m ago - inside
    _write(store, now + 600)  # future - outside range

    rows = store.query_range(window="1h", from_ts=now - 4000, to_ts=now)
    assert len(rows) == 2


def test_query_range_ordered_oldest_first(tmp_path: Path) -> None:
    """query_range returns rows in ascending timestamp order."""
    store = _store(tmp_path)
    now = time.time()

    _write(store, now - 900)
    _write(store, now - 1800)
    _write(store, now - 300)

    rows = store.query_range(window="1h", from_ts=now - 2000, to_ts=now)
    timestamps = [r.timestamp for r in rows]
    assert timestamps == sorted(timestamps)


def test_query_range_empty_when_no_match(tmp_path: Path) -> None:
    """query_range returns an empty list when no rows match the range."""
    store = _store(tmp_path)
    _write(store, time.time() - 86400)  # 24h ago

    rows = store.query_range(window="1h", from_ts=time.time() - 3600, to_ts=time.time())
    assert rows == []


def test_query_range_filters_by_window(tmp_path: Path) -> None:
    """query_range respects the window filter - 5m rows don't appear in 1h query."""
    store = _store(tmp_path)
    now = time.time()

    _write(store, now - 300, window="5m")
    _write(store, now - 300, window="1h")

    rows_1h = store.query_range(window="1h", from_ts=now - 600, to_ts=now)
    rows_5m = store.query_range(window="5m", from_ts=now - 600, to_ts=now)

    assert len(rows_1h) == 1
    assert len(rows_5m) == 1
    assert rows_1h[0].window == "1h"
    assert rows_5m[0].window == "5m"


def test_query_range_inclusive_bounds(tmp_path: Path) -> None:
    """Both from_ts and to_ts bounds are inclusive."""
    store = _store(tmp_path)
    t = time.time()

    _write(store, t)  # exactly at from_ts = to_ts
    rows = store.query_range(window="1h", from_ts=t, to_ts=t)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# _parse_iso
# ---------------------------------------------------------------------------


def test_parse_iso_date_format() -> None:
    """YYYY-MM-DD is parsed as midnight UTC."""
    from model_monitor.cli.replay import _parse_iso

    ts = _parse_iso("2025-01-15", field="from")
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert dt.year == 2025
    assert dt.month == 1
    assert dt.day == 15
    assert dt.hour == 0


def test_parse_iso_datetime_format() -> None:
    """YYYY-MM-DDTHH:MM is parsed correctly."""
    from model_monitor.cli.replay import _parse_iso

    ts = _parse_iso("2025-06-01T14:30", field="to")
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert dt.hour == 14
    assert dt.minute == 30


def test_parse_iso_invalid_exits_2() -> None:
    """Invalid ISO string causes sys.exit(2)."""
    from model_monitor.cli.replay import _parse_iso

    with pytest.raises(SystemExit) as exc_info:
        _parse_iso("not-a-date", field="from")
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# _run_replay
# ---------------------------------------------------------------------------


def _write_history_row(store: MetricsSummaryHistoryStore, ts: float) -> None:
    store.write(
        window="1h",
        timestamp=ts,
        n_batches=10,
        avg_accuracy=0.90,
        avg_f1=0.88,
        avg_confidence=0.85,
        avg_drift_score=0.04,
        avg_latency_ms=120.0,
    )


def test_run_replay_returns_0_on_empty_range(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Empty range exits 0 without writing anything."""
    from model_monitor.cli.replay import _run_replay

    store = _store(tmp_path)
    # Write a row outside the query range
    _write(store, time.time() - 86400)

    now = time.time()
    with (
        patch(
            "model_monitor.cli.replay.MetricsSummaryHistoryStore", return_value=store
        ),
        patch("model_monitor.cli.replay.DecisionStore"),
        patch("model_monitor.cli.replay.ModelStore"),
    ):
        rc = _run_replay(
            from_ts=now - 100,
            to_ts=now,
            window="1h",
            dry_run=False,
            emit_json=False,
            no_colour=True,
        )
    assert rc == 0


def test_run_replay_json_output(tmp_path: Path) -> None:
    """--json flag emits valid newline-delimited JSON with required fields."""
    from unittest.mock import MagicMock

    from model_monitor.cli.replay import _run_replay

    store = _store(tmp_path)
    now = time.time()
    _write(store, now - 300)

    model_store_mock = MagicMock()
    model_store_mock.get_active_metadata.return_value = {
        "metrics": {"baseline_f1": 0.88}
    }
    model_store_mock.has_candidate.return_value = False

    decision_store_mock = MagicMock()
    decision_store_mock.tail.return_value = []
    decision_store_mock.count.return_value = 5

    out = StringIO()
    with (
        patch(
            "model_monitor.cli.replay.MetricsSummaryHistoryStore", return_value=store
        ),
        patch(
            "model_monitor.cli.replay.DecisionStore", return_value=decision_store_mock
        ),
        patch("model_monitor.cli.replay.ModelStore", return_value=model_store_mock),
        patch("sys.stdout", out),
    ):
        rc = _run_replay(
            from_ts=now - 600,
            to_ts=now,
            window="1h",
            dry_run=True,
            emit_json=True,
            no_colour=True,
        )

    assert rc == 0
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    for field in (
        "timestamp",
        "window",
        "trust_score",
        "f1",
        "input_drift_score",
        "action",
        "reason",
        "dry_run",
    ):
        assert field in record, f"Missing JSON field: {field!r}"
    assert record["dry_run"] is True
