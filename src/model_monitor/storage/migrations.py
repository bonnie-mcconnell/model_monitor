"""Schema migration runner for the model_monitor SQLite database.

Uses a lightweight schema_version table to track applied migrations
rather than a full Alembic setup, keeping the dependency footprint
minimal while still providing safe, auditable schema evolution.

Why not Alembic
---------------
Alembic is the right tool when you need multi-database support, autogenerate
from ORM diffs, or complex branching migrations.  model_monitor uses a single
SQLite file, has a small and stable schema, and deploys as a single process.
Alembic would add a dependency, a migration directory, and `alembic.ini` for
a problem that five append-only Python functions solve completely.  If the
schema grows substantially or the storage backend changes to Postgres, migrating
to Alembic from this pattern is straightforward - the version tracking concept
is identical.

Design
------
Migrations are append-only Python functions keyed by an integer version.
On startup, ``run_migrations(engine)`` applies any unapplied migrations
in order.  The highest applied version is stored in ``schema_version``.

Adding a migration
------------------
1. Define a function ``_migration_N(conn)`` that executes the DDL.
2. Add it to ``_MIGRATIONS`` at position N-1 (zero-indexed).
3. The next server restart applies it automatically.

Thread safety
-------------
Migrations run inside a serialised DDL transaction.  Concurrent server
instances (e.g. during a rolling restart) are safe because SQLite's WAL
mode serialises writes and the version check is inside the same
transaction as the DDL.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import Connection, inspect, text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Migration functions
# ---------------------------------------------------------------------------
# Each function receives a raw DBAPI connection.  Use conn.execute(text(...))
# so SQLAlchemy's text() wrapper handles quoting correctly.


def _migration_1(conn: Connection) -> None:
    """Baseline: create schema_version table.

    All other tables are created by SQLAlchemy's metadata.create_all() on
    startup; this migration only installs the version-tracking table itself.
    The table stores a single row because SQLite has no per-schema metadata.
    """
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL,
                applied_at REAL NOT NULL
            )
            """
        )
    )


def _migration_2(conn: Connection) -> None:
    """Add behavioral_violation_rate to metrics table if missing.

    This column was added in a later feature pass.  Existing databases that
    were created before the column existed need an ALTER TABLE; new databases
    get it from create_all() and this migration is a no-op (SQLite silently
    accepts duplicate column additions when wrapped in a try/except).
    """
    insp = inspect(conn)
    cols = {c["name"] for c in insp.get_columns("metrics")}
    if "behavioral_violation_rate" not in cols:
        conn.execute(
            text("ALTER TABLE metrics ADD COLUMN behavioral_violation_rate REAL")
        )


def _migration_3(conn: Connection) -> None:
    """Add shap_attribution column to metrics table if missing."""
    insp = inspect(conn)
    cols = {c["name"] for c in insp.get_columns("metrics")}
    if "shap_attribution" not in cols:
        conn.execute(text("ALTER TABLE metrics ADD COLUMN shap_attribution TEXT"))


def _migration_4(conn: Connection) -> None:
    """Add calibration_error column to metrics table if missing."""
    insp = inspect(conn)
    cols = {c["name"] for c in insp.get_columns("metrics")}
    if "calibration_error" not in cols:
        conn.execute(text("ALTER TABLE metrics ADD COLUMN calibration_error REAL"))


def _migration_5(conn: Connection) -> None:
    """Add feature_drift_scores column to metrics table if missing."""
    insp = inspect(conn)
    cols = {c["name"] for c in insp.get_columns("metrics")}
    if "feature_drift_scores" not in cols:
        conn.execute(text("ALTER TABLE metrics ADD COLUMN feature_drift_scores TEXT"))


# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------
# Zero-indexed list.  _MIGRATIONS[0] is migration version 1.
def _migration_6(conn: Connection) -> None:
    """Add p95_latency_ms and p99_latency_ms columns to metrics table."""
    insp = inspect(conn)
    cols = {c["name"] for c in insp.get_columns("metrics")}
    if "p95_latency_ms" not in cols:
        conn.execute(text("ALTER TABLE metrics ADD COLUMN p95_latency_ms REAL"))
    if "p99_latency_ms" not in cols:
        conn.execute(text("ALTER TABLE metrics ADD COLUMN p99_latency_ms REAL"))


def _migration_7(conn: Connection) -> None:
    """Add output drift, data quality, and conformal columns to metrics table."""
    insp = inspect(conn)
    cols = {c["name"] for c in insp.get_columns("metrics")}
    additions = {
        "output_drift_score": "REAL",
        "output_drift_class_scores": "TEXT",
        "data_quality_score": "REAL",
        "conformal_coverage": "REAL",
        "conformal_set_size": "REAL",
    }
    for col_name, col_type in additions.items():
        if col_name not in cols:
            conn.execute(text(f"ALTER TABLE metrics ADD COLUMN {col_name} {col_type}"))


def _migration_8(conn: Connection) -> None:
    """Add new monitoring signal columns to metrics_summary table."""
    insp = inspect(conn)
    cols = {col["name"] for col in insp.get_columns("metrics_summary")}
    additions = {
        "avg_output_drift_score": "REAL",
        "avg_data_quality_score": "REAL",
        "avg_conformal_coverage": "REAL",
        "avg_conformal_set_size": "REAL",
    }
    for col_name, col_type in additions.items():
        if col_name not in cols:
            conn.execute(
                text(f"ALTER TABLE metrics_summary ADD COLUMN {col_name} {col_type}")
            )


def _migration_9(conn: Connection) -> None:
    """Add new monitoring signal columns to metrics_summary_history table."""
    insp = inspect(conn)
    cols = {col["name"] for col in insp.get_columns("metrics_summary_history")}
    additions = {
        "avg_output_drift_score": "REAL",
        "avg_data_quality_score": "REAL",
        "avg_conformal_coverage": "REAL",
        "avg_conformal_set_size": "REAL",
    }
    for col_name, col_type in additions.items():
        if col_name not in cols:
            conn.execute(
                text(
                    f"ALTER TABLE metrics_summary_history ADD COLUMN {col_name} {col_type}"
                )
            )


_MIGRATIONS: list[Callable[[Connection], None]] = [
    _migration_1,
    _migration_2,
    _migration_3,
    _migration_4,
    _migration_5,
    _migration_6,
    _migration_7,
    _migration_8,
    _migration_9,
]

CURRENT_SCHEMA_VERSION: int = len(_MIGRATIONS)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _get_schema_version(conn: Connection) -> int:
    """Return the current schema version, or 0 if the table does not exist."""
    try:
        row = conn.execute(text("SELECT MAX(version) FROM schema_version")).fetchone()
        if row is None or row[0] is None:
            return 0
        return int(row[0])
    except Exception:
        return 0


def run_migrations(engine: Any) -> None:
    """Apply all pending migrations to ``engine`` in order.

    Safe to call on every startup.  Already-applied migrations are skipped.
    Each migration runs in its own transaction; a failure leaves the database
    at the last successfully applied version rather than in a partial state.

    Args:
        engine: SQLAlchemy engine connected to the metrics database.
    """
    import time

    with engine.begin() as conn:
        current = _get_schema_version(conn)

    if current >= CURRENT_SCHEMA_VERSION:
        log.debug(
            "schema_up_to_date",
            extra={
                "version": current,
                "latest": CURRENT_SCHEMA_VERSION,
            },
        )
        return

    log.info(
        "schema_migration_start",
        extra={
            "from_version": current,
            "to_version": CURRENT_SCHEMA_VERSION,
        },
    )

    for i, migration_fn in enumerate(_MIGRATIONS):
        version = i + 1
        if version <= current:
            continue

        with engine.begin() as conn:
            try:
                migration_fn(conn)
                conn.execute(
                    text(
                        "INSERT INTO schema_version (version, applied_at) "
                        "VALUES (:v, :ts)"
                    ),
                    {"v": version, "ts": time.time()},
                )
                log.info(
                    "schema_migration_applied",
                    extra={"version": version, "migration": migration_fn.__name__},
                )
            except Exception:
                log.exception(
                    "schema_migration_failed",
                    extra={"version": version, "migration": migration_fn.__name__},
                )
                raise
