from __future__ import annotations

import logging
from typing import Mapping

from .thresholds import CRITICAL_TRUST_SCORE, MIN_TRUST_SCORE

logger = logging.getLogger("model_monitor.alerts")


def check_alerts(window: str, summary: Mapping[str, float]) -> None:
    trust = summary.get("trust_score")
    if trust is None:
        return

    if trust < CRITICAL_TRUST_SCORE:
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
        logger.warning(
            "Trust score below operational floor",
            extra={
                "window": window,
                "trust_score": trust,
                "severity": "warning",
            },
        )
