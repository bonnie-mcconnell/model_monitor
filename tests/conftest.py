"""Shared pytest fixtures for model_monitor tests.

Fixtures defined here are available to every test file without import.
Keep this file focused on infrastructure - store factories, tmp-path
helpers, and session-level setup.  Test-specific helpers belong in the
test file that uses them.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from model_monitor.storage.alert_store import AlertStore
from model_monitor.storage.db import Base, engine
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.model_store import ModelStore

# ---------------------------------------------------------------------------
# Session-level table setup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def create_test_tables() -> Generator[None, None, None]:
    """Create all database tables once for the full test session.

    drop_all on teardown ensures a clean state if the test database
    is reused between runs (e.g. the SQLite file persists on disk).
    """
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


# ---------------------------------------------------------------------------
# Isolated store factories (function scope - fresh database per test)
# ---------------------------------------------------------------------------


@pytest.fixture()
def metrics_store(tmp_path: Path) -> MetricsStore:
    """Return a MetricsStore backed by an isolated per-test SQLite file."""
    return MetricsStore(db_path=tmp_path / "metrics.db")


@pytest.fixture()
def decision_store(tmp_path: Path) -> DecisionStore:
    """Return a DecisionStore backed by an isolated per-test SQLite file."""
    return DecisionStore(db_path=tmp_path / "decisions.db")


@pytest.fixture()
def alert_store(tmp_path: Path) -> AlertStore:
    """Return an AlertStore backed by an isolated per-test SQLite file."""
    return AlertStore(db_path=tmp_path / "alerts.db")


@pytest.fixture()
def model_store(tmp_path: Path) -> ModelStore:
    """Return a ModelStore backed by an isolated per-test directory."""
    store_dir = tmp_path / "models"
    store_dir.mkdir()
    return ModelStore(base_path=str(store_dir))
