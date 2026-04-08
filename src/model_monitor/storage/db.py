"""SQLAlchemy engine and session factory shared by all storage modules."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# ---------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------

DATABASE_URL = "sqlite:///data/metrics/metrics.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite concurrency
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    class_=Session,
)

Base = declarative_base()

# ---------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------

@contextmanager
def get_session() -> Iterator[Session]:
    """
    Context-managed database session.

    Commits on success, rolls back on exception.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
