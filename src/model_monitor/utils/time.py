"""UTC timestamp helpers."""
from datetime import datetime, timezone


def utc_now():
    """
    Return current UTC time as a timezone-aware datetime.
    """
    return datetime.now(timezone.utc)


def utc_iso():
    """
    Return current UTC time in ISO 8601 format.
    """
    return utc_now().isoformat()
