"""
Canonical aggregation window definitions.

Single source of truth for the set of time windows used across the
monitoring system. Any code that needs to enumerate or compute over
windows imports from here - never hard-codes "5m", "1h", "24h".

Adding a new window (e.g. "7d") means changing this file only.
"""
from __future__ import annotations

AGGREGATION_WINDOWS: dict[str, int] = {
    "5m":  5 * 60,
    "1h":  60 * 60,
    "24h": 24 * 60 * 60,
}
