"""UTC timestamp helpers."""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return utc_now().isoformat()
