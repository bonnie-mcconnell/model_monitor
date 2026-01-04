from __future__ import annotations

import logging
import time
from typing import Mapping

from .thresholds import CRITICAL_TRUST_SCORE, MIN_TRUST_SCORE

logger = logging.getLogger("model_monitor.alerts")

# Basic in-process alert suppression
_last_alert_ts: dict[str, float] = {}
_ALERT_COOLDOWN_SECONDS = 300  # 5 minutes


def _can_emit(key: str) -> bool:
    now = time.time()
    last = _last_alert_ts.get(key, 0.0)
    if now - last < _ALERT_COOLDOWN_SECONDS:
        return False
    _last_alert_ts[key] = now
    return True


def check_alerts(window: str, summary: Mapping[str, float]) -> None:
    """
    Emit alerts based on trust score thresholds.

    This function:
    - logs alerts only
    - performs no state mutation
    - does not trigger decisions
    """
    trust = summary.get("trust_score")
    if trust is None:
        return

    if trust < CRITICAL_TRUST_SCORE:
        if _can_emit(f"{window}:critical"):
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
        if _can_emit(f"{window}:warning"):
            logger.warning(
                "Trust score below operational floor",
                extra={
                    "window": window,
                    "trust_score": trust,
                    "severity": "warning",
                },
            )
