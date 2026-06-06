"""Persistence layer: SQLite-backed stores for metrics, decisions, and models.

All stores share the engine and session factory from db.py. Each store
exposes a narrow, typed interface; no SQL escapes into application code.
"""

from __future__ import annotations
