from __future__ import annotations

import logging

logger = logging.getLogger("model_monitor.alerts")

TRUST_THRESHOLDS = {
    "5m": 0.65,
    "1h": 0.70,
    "24h": 0.75,
}


def check_alerts(window: str, summary: dict) -> None:
    trust = summary["trust_score"]
    threshold = TRUST_THRESHOLDS.get(window)

    if threshold is None:
        return

    if trust < threshold:
        logger.warning(
            "Trust score below threshold",
            extra={
                "window": window,
                "trust_score": trust,
                "threshold": threshold,
            },
        )
