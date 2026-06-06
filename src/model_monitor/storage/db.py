"""SQLAlchemy engine and session factory shared by all storage modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Override the default path by setting DATABASE_URL in your environment.
# Useful for running the server from a non-repo-root directory or for
# pointing multiple environments at different databases without code changes.
#
# Examples:
#   DATABASE_URL=sqlite:////absolute/path/to/metrics.db uvicorn ...
#   DATABASE_URL=sqlite:///./data/metrics/metrics.db uvicorn ...
_DEFAULT_DATABASE_URL = "sqlite:///data/metrics/metrics.db"
DATABASE_URL: str = os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)


def _ensure_database_directory(url: str) -> None:
    """Create the directory containing the SQLite file if it does not exist.

    SQLite will create the database file itself, but it will not create missing
    parent directories - the open() call fails with OperationalError on a clean
    checkout where data/metrics/ was never committed.  This function extracts
    the filesystem path from a sqlite:/// URL and calls mkdir() so the engine
    can always connect without manual setup steps.

    In-memory databases (sqlite:///:memory:) and non-SQLite URLs are left
    unchanged.
    """
    if not url.startswith("sqlite:///"):
        return
    # sqlite:///relative/path  →  relative/path
    # sqlite:////absolute/path →  /absolute/path
    db_path_str = url[len("sqlite:///"):]
    if db_path_str in (":memory:", ""):
        return
    Path(db_path_str).parent.mkdir(parents=True, exist_ok=True)


_ensure_database_directory(DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite concurrency
    future=True,
)


def _set_sqlite_pragmas(dbapi_conn: Any, _connection_record: Any) -> None:
    """Configure SQLite for concurrent read-write workloads.

    WAL (Write-Ahead Logging) mode allows readers to proceed concurrently with
    a single writer - critical because the aggregation loop writes summaries
    while the ingest API writes metric records.  Without WAL, every writer
    holds an exclusive lock and readers block until the write completes.

    NORMAL synchronous mode syncs at the most critical moments only (WAL
    checkpoint), giving a significant throughput improvement over the default
    FULL mode with negligible durability trade-off on typical hardware.

    These are connection-level settings in SQLite, so they must be set on
    every new connection via this event hook rather than once at engine
    creation time.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


# Register the PRAGMA hook for every new connection made by this engine.
# This fires before the connection is handed to SQLAlchemy's connection pool,
# so every session that uses this engine gets WAL mode automatically.
event.listen(engine, "connect", _set_sqlite_pragmas)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    class_=Session,
)

Base = declarative_base()
