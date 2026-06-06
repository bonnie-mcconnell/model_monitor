"""Tests for storage/migrations.py.

Verifies:
- Migrations apply in order on a fresh database.
- Re-running migrations on an already-migrated database is a no-op.
- The schema_version table records every applied migration.
- Individual migration functions are idempotent (column already exists → no crash).
- A partial migration state (version N) is correctly resumed from N+1.
- CURRENT_SCHEMA_VERSION matches the length of the migration list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text

from model_monitor.storage.db import Base
from model_monitor.storage.migrations import (
    _MIGRATIONS,
    CURRENT_SCHEMA_VERSION,
    _get_schema_version,
    run_migrations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(tmp_path: Path) -> Any:
    """Return a fresh SQLite engine backed by a temp file."""
    db_path = tmp_path / "test_migrations.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _applied_versions(engine: Any) -> list[int]:
    """Return all applied migration version numbers from schema_version."""
    with engine.connect() as conn:
        try:
            rows = conn.execute(
                text("SELECT version FROM schema_version ORDER BY version")
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------


def test_run_migrations_on_fresh_db_applies_all(tmp_path: Path) -> None:
    """All migrations are applied once on a database that has never been migrated."""
    engine = _make_engine(tmp_path)
    run_migrations(engine)

    versions = _applied_versions(engine)
    assert versions == list(range(1, CURRENT_SCHEMA_VERSION + 1)), (
        f"Expected versions 1–{CURRENT_SCHEMA_VERSION}, got {versions}"
    )


def test_run_migrations_idempotent(tmp_path: Path) -> None:
    """Calling run_migrations() twice on the same database is a no-op.

    The second call must not raise and must not insert duplicate rows into
    schema_version - which would cause confusing output in operational logs.
    """
    engine = _make_engine(tmp_path)
    run_migrations(engine)
    run_migrations(engine)  # second call - must be silent

    versions = _applied_versions(engine)
    assert versions == list(range(1, CURRENT_SCHEMA_VERSION + 1)), (
        "Idempotency violation: duplicate version rows after second run_migrations()"
    )


def test_schema_version_table_created_by_migration_1(tmp_path: Path) -> None:
    """Migration 1 creates the schema_version table; version 0 before it runs."""
    engine = _make_engine(tmp_path)

    with engine.connect() as conn:
        before = _get_schema_version(conn)

    assert before == 0, "Fresh database should report version 0"

    run_migrations(engine)

    with engine.connect() as conn:
        after = _get_schema_version(conn)

    assert after == CURRENT_SCHEMA_VERSION


def test_migration_adds_columns_to_metrics_table(tmp_path: Path) -> None:
    """Migrations 2–5 add expected columns to the metrics table."""
    engine = _make_engine(tmp_path)
    run_migrations(engine)

    with engine.connect() as conn:
        cols = {c["name"] for c in inspect(conn).get_columns("metrics")}

    expected = {
        "behavioral_violation_rate",
        "shap_attribution",
        "calibration_error",
        "feature_drift_scores",
    }
    missing = expected - cols
    assert not missing, f"Columns missing after migrations: {missing}"


def test_column_addition_migrations_idempotent_when_column_exists(
    tmp_path: Path,
) -> None:
    """Column-addition migrations do not crash if the column already exists.

    This simulates a database that was created from a schema that already
    included the column (e.g. via create_all() on a current ORM model) and
    then has run_migrations() applied to it.  The ALTER TABLE branch checks
    for the column first and skips the DDL if present.
    """
    engine = _make_engine(tmp_path)
    # Run once to set up schema_version and add all columns.
    run_migrations(engine)

    # Run individual migration functions directly against the migrated db.
    # They must not raise even though the columns are already there.
    from model_monitor.storage.migrations import (
        _migration_2,
        _migration_3,
        _migration_4,
        _migration_5,
    )

    with engine.begin() as conn:
        _migration_2(conn)
        _migration_3(conn)
        _migration_4(conn)
        _migration_5(conn)


def test_partial_migration_resumes_from_correct_version(tmp_path: Path) -> None:
    """A database at version N is brought to CURRENT without re-running 1..N.

    Simulates a server that was stopped mid-migration or deployed from an
    older image, then upgraded.  Only the missing migrations must fire.
    """
    engine = _make_engine(tmp_path)

    # Manually apply only migration 1 (creates schema_version).
    import time

    with engine.begin() as conn:
        _MIGRATIONS[0](conn)
        conn.execute(
            text("INSERT INTO schema_version (version, applied_at) VALUES (1, :ts)"),
            {"ts": time.time()},
        )

    with engine.connect() as conn:
        assert _get_schema_version(conn) == 1

    # Now run_migrations should apply 2..CURRENT only.
    run_migrations(engine)

    versions = _applied_versions(engine)
    assert versions == list(range(1, CURRENT_SCHEMA_VERSION + 1))


# ---------------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------------


def test_current_schema_version_matches_migration_list_length() -> None:
    """CURRENT_SCHEMA_VERSION must equal len(_MIGRATIONS).

    Prevents the common mistake of adding a migration function without
    updating the sentinel, which would cause silent under-migration.
    """
    assert CURRENT_SCHEMA_VERSION == len(_MIGRATIONS), (
        f"CURRENT_SCHEMA_VERSION={CURRENT_SCHEMA_VERSION} but "
        f"len(_MIGRATIONS)={len(_MIGRATIONS)}. Update CURRENT_SCHEMA_VERSION "
        "or add/remove a migration function."
    )


def test_metrics_store_runs_migrations_on_init(tmp_path: Path) -> None:
    """MetricsStore.__init__ must run migrations so callers get a current schema."""
    from model_monitor.storage.metrics_store import MetricsStore

    db_path = tmp_path / "metrics.db"
    MetricsStore(db_path=db_path)

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    with engine.connect() as conn:
        version = _get_schema_version(conn)

    assert version == CURRENT_SCHEMA_VERSION, (
        f"MetricsStore.__init__ did not run migrations; "
        f"schema at version {version}, expected {CURRENT_SCHEMA_VERSION}"
    )
