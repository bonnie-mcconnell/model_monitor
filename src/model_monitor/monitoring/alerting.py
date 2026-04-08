"""In-process alerting on trust score thresholds with per-key cooldown suppression."""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping

from .thresholds import CRITICAL_TRUST_SCORE, MIN_TRUST_SCORE

logger = logging.getLogger("model_monitor.alerts")
logger.propagate = True

_ALERT_COOLDOWN_SECONDS: int = 300  # 5 minutes


class AlertCooldownTracker:
    """
    Tracks per-key alert cooldowns so repeated alerts are suppressed.

    A separate class rather than a module-level dict for two reasons:
    - Tests can instantiate a fresh tracker without touching module state
    - The tracker is independently injectable into any alerting path

    Each key is a ``(window, severity)`` string. An alert is allowed at
    most once per ``cooldown_seconds`` for each key.
    """

    def __init__(self, cooldown_seconds: int = _ALERT_COOLDOWN_SECONDS) -> None:
        self._cooldown = cooldown_seconds
        self._last_ts: dict[str, float] = {}

    def can_emit(self, key: str) -> bool:
        """Return True and record the timestamp if the cooldown has elapsed."""
        now = time.time()
        if now - self._last_ts.get(key, 0.0) < self._cooldown:
            return False
        self._last_ts[key] = now
        return True

    def reset(self) -> None:
        """Clear all cooldown state. Intended for tests only."""
        self._last_ts.clear()


# Process-level singleton used by check_alerts().
# Tests that need isolation should call check_alerts() via a fresh
# AlertCooldownTracker and pass it explicitly — or reset this one via
# the autouse fixture in test_alerting.py.
_default_tracker = AlertCooldownTracker()


def check_alerts(
    window: str,
    summary: Mapping[str, float],
    *,
    tracker: AlertCooldownTracker | None = None,
) -> None:
    """
    Emit structured log alerts when trust score crosses operational thresholds.

    Behaviour:
    - Logs only; never mutates state, never triggers decisions
    - Each (window, severity) pair is suppressed for ``cooldown_seconds``
      after firing to prevent alert fatigue on persistent degradation

    Args:
        window:  aggregation window that produced this summary (e.g. "5m")
        summary: mapping that must contain a ``trust_score`` float key
        tracker: cooldown tracker to use; defaults to the process singleton.
                 Pass a fresh ``AlertCooldownTracker()`` in tests that need
                 full isolation without touching module-level state.
    """
    t = tracker if tracker is not None else _default_tracker

    trust = summary.get("trust_score")
    if trust is None:
        return

    if trust < CRITICAL_TRUST_SCORE:
        if t.can_emit(f"{window}:critical"):
            logger.error(
                "Critical trust degradation detected",
                extra={
                    "window": window,
                    "trust_score": trust,
                    "severity": "critical",
                },
            )
        return

    if trust < MIN_TRUST_SCORE:
        if t.can_emit(f"{window}:warning"):
            logger.warning(
                "Trust score below operational floor",
                extra={
                    "window": window,
                    "trust_score": trust,
                    "severity": "warning",
                },
            )
