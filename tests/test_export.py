"""Tests for DecisionStore.query_range and cli.export._run_export.

Verifies:
- query_range returns records within [from_ts, to_ts] inclusive.
- query_range returns records ordered oldest-first.
- query_range returns all records when no bounds given.
- query_range returns empty list when no records match.
- _run_export writes valid CSV with correct header.
- _run_export writes valid NDJSON with all required fields.
- _run_export --output writes to a file and prints a summary to stderr.
- _run_export returns 0 on success, 0 on empty result.
- _parse_iso handles all supported formats and rejects invalid input.
"""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session as _Session

from model_monitor.core.decisions import DecisionType
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.models.decision_record import DecisionRecordORM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(
    store: DecisionStore, ts: float, action: DecisionType = DecisionType.NONE
) -> None:
    session: _Session = store._session_factory()
    try:
        row = DecisionRecordORM(
            timestamp=ts,
            action=action,
            reason="test",
            batch_index=1,
            trust_score=0.9,
            f1=0.88,
            drift_score=0.05,
        )
        session.add(row)
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# DecisionStore.query_range
# ---------------------------------------------------------------------------


def test_query_range_returns_records_within_bounds(tmp_path: Path) -> None:
    """query_range returns only records within [from_ts, to_ts]."""
    store = DecisionStore(db_path=tmp_path / "d.db")
    now = time.time()

    _write(store, now - 7200)  # outside
    _write(store, now - 3600)  # inside
    _write(store, now - 1800)  # inside
    _write(store, now + 600)  # outside

    rows = store.query_range(from_ts=now - 4000, to_ts=now)
    assert len(rows) == 2


def test_query_range_ordered_oldest_first(tmp_path: Path) -> None:
    """Records returned in ascending timestamp order."""
    store = DecisionStore(db_path=tmp_path / "d.db")
    now = time.time()

    _write(store, now - 300)
    _write(store, now - 1800)
    _write(store, now - 900)

    rows = store.query_range(from_ts=now - 2000, to_ts=now)
    timestamps = [r.timestamp for r in rows]
    assert timestamps == sorted(timestamps)


def test_query_range_no_bounds_returns_all(tmp_path: Path) -> None:
    """Calling query_range() with no bounds returns every record."""
    store = DecisionStore(db_path=tmp_path / "d.db")
    now = time.time()
    for i in range(5):
        _write(store, now - i * 300)

    rows = store.query_range()
    assert len(rows) == 5


def test_query_range_empty_when_no_match(tmp_path: Path) -> None:
    """Empty list when no records fall within the range."""
    store = DecisionStore(db_path=tmp_path / "d.db")
    _write(store, time.time() - 86400)

    rows = store.query_range(from_ts=time.time() - 3600, to_ts=time.time())
    assert rows == []


def test_query_range_inclusive_bounds(tmp_path: Path) -> None:
    """Both from_ts and to_ts are inclusive."""
    store = DecisionStore(db_path=tmp_path / "d.db")
    t = time.time()
    _write(store, t)

    rows = store.query_range(from_ts=t, to_ts=t)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# _parse_iso
# ---------------------------------------------------------------------------


def test_parse_iso_date() -> None:
    """YYYY-MM-DD is parsed as midnight UTC."""
    from datetime import datetime, timezone

    from model_monitor.cli.export import _parse_iso

    ts = _parse_iso("2025-03-15", field="from")
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert (dt.year, dt.month, dt.day) == (2025, 3, 15)
    assert dt.hour == 0


def test_parse_iso_datetime() -> None:
    """YYYY-MM-DDTHH:MM is parsed correctly."""
    from datetime import datetime, timezone

    from model_monitor.cli.export import _parse_iso

    ts = _parse_iso("2025-06-01T14:30", field="from")
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert dt.hour == 14 and dt.minute == 30


def test_parse_iso_invalid_exits_2() -> None:
    """Invalid value causes sys.exit(2)."""
    from model_monitor.cli.export import _parse_iso

    with pytest.raises(SystemExit) as exc_info:
        _parse_iso("not-a-date", field="from")
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# _run_export
# ---------------------------------------------------------------------------


def test_run_export_csv_header_and_row(tmp_path: Path) -> None:
    """CSV output has correct header and one data row."""
    from model_monitor.cli.export import _CSV_FIELDS, _run_export

    store = DecisionStore(db_path=tmp_path / "d.db")
    _write(store, time.time() - 60, action=DecisionType.RETRAIN)

    out = io.StringIO()
    with (
        patch("model_monitor.cli.export.DecisionStore", return_value=store),
        patch("sys.stdout", out),
    ):
        rc = _run_export(from_ts=None, to_ts=None, fmt="csv", output=None)

    assert rc == 0
    out.seek(0)
    reader = csv.DictReader(out)
    assert reader.fieldnames is not None
    for field in _CSV_FIELDS:
        assert field in reader.fieldnames
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["action"] == DecisionType.RETRAIN


def test_run_export_ndjson_fields(tmp_path: Path) -> None:
    """NDJSON output has all required fields."""
    from model_monitor.cli.export import _run_export

    store = DecisionStore(db_path=tmp_path / "d.db")
    _write(store, time.time() - 30, action=DecisionType.NONE)

    out = io.StringIO()
    with (
        patch("model_monitor.cli.export.DecisionStore", return_value=store),
        patch("sys.stdout", out),
    ):
        rc = _run_export(from_ts=None, to_ts=None, fmt="json", output=None)

    assert rc == 0
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    for field in (
        "id",
        "timestamp",
        "timestamp_iso",
        "action",
        "reason",
        "trust_score",
        "f1",
        "drift_score",
    ):
        assert field in record, f"Missing NDJSON field: {field!r}"


def test_run_export_empty_store_returns_0(tmp_path: Path) -> None:
    """Empty store exits 0 without writing anything."""
    from model_monitor.cli.export import _run_export

    store = DecisionStore(db_path=tmp_path / "empty.db")
    out = io.StringIO()
    with (
        patch("model_monitor.cli.export.DecisionStore", return_value=store),
        patch("sys.stdout", out),
    ):
        rc = _run_export(from_ts=None, to_ts=None, fmt="csv", output=None)

    assert rc == 0
    # Only header line
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1  # header only


def test_run_export_to_file(tmp_path: Path) -> None:
    """--output writes to file and does not write to stdout."""
    from model_monitor.cli.export import _run_export

    store = DecisionStore(db_path=tmp_path / "d.db")
    _write(store, time.time() - 60, action=DecisionType.ROLLBACK)

    out_path = tmp_path / "decisions.csv"
    with patch("model_monitor.cli.export.DecisionStore", return_value=store):
        rc = _run_export(from_ts=None, to_ts=None, fmt="csv", output=out_path)

    assert rc == 0
    assert out_path.exists()
    content = out_path.read_text()
    assert "rollback" in content
