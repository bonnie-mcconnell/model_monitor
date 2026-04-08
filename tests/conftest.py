from __future__ import annotations

# tests/conftest.py
from collections.abc import Generator

import pytest

# Explicit import so Base.metadata discovers every table before create_all.
# Without this, tables added after the initial conftest was written would
# silently not be created, causing tests to fail with "no such table" errors.
import model_monitor.storage.models  # noqa: F401
from model_monitor.storage.db import Base, engine


@pytest.fixture(scope="session", autouse=True)
def create_test_tables() -> Generator[None, None, None]:
    """
    Create all database tables once for the full test session.

    drop_all on teardown ensures a clean state if the test database
    is reused between runs (e.g. the SQLite file persists on disk).
    """
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
