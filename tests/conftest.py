# tests/conftest.py
import pytest

from model_monitor.storage.db import engine
from model_monitor.storage.db import Base


@pytest.fixture(scope="session", autouse=True)
def create_test_tables():
    """
    Create all database tables for tests.
    """
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
