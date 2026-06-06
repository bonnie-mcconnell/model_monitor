"""In-process alerting on trust score thresholds with per-key cooldown suppression."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

try:
    import requests as _requests_lib

    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from .thresholds import (
    CRITICAL_TRUST_SCORE,
    MAX_OUTPUT_DRIFT_SCORE,
    MIN_CONFORMAL_COVERAGE,
    MIN_DATA_QUALITY_SCORE,
    MIN_TRUST_SCORE,
)

if TYPE_CHECKING:
    from model_monitor.storage.alert_store import AlertStore

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
# AlertCooldownTracker and pass it explicitly - or reset this one via
# the autouse fixture in test_alerting.py.
_default_tracker = AlertCooldownTracker()


def check_alerts(
    window: str,
    summary: Mapping[str, float],
    *,
    tracker: AlertCooldownTracker | None = None,
    alert_store: AlertStore | None = None,
) -> None:
    """
    Emit structured log alerts when trust score crosses operational thresholds.

    Behaviour:
    - Logs at WARNING or ERROR level when trust score crosses thresholds
    - Each (window, severity) pair is suppressed for ``cooldown_seconds``
      after firing to prevent alert fatigue on persistent degradation
    - When ``alert_store`` is supplied, every fired alert is also persisted
      so operators can query alert history via the API

    Args:
        window:      aggregation window label (e.g. "5m")
        summary:     mapping that must contain a ``trust_score`` float key
        tracker:     cooldown tracker; defaults to the process singleton
        alert_store: optional persistence layer for alert history
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
            if alert_store is not None:
                try:
                    alert_store.record(
                        window=window,
                        severity="critical",
                        trust_score=float(trust),
                    )
                except Exception:
                    logger.warning("alert_store_write_failed", exc_info=True)
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
            if alert_store is not None:
                try:
                    alert_store.record(
                        window=window,
                        severity="warning",
                        trust_score=float(trust),
                    )
                except Exception:
                    logger.warning("alert_store_write_failed", exc_info=True)

    # ── Conformal coverage alert ───────────────────────────────────────────
    # A drop below MIN_CONFORMAL_COVERAGE is a provable signal: the model's
    # coverage guarantee is breaking down.  This is rigorous (not heuristic)
    # and warrants immediate operator attention even when trust_score is OK.
    coverage = summary.get("avg_conformal_coverage")
    if coverage is not None and coverage < MIN_CONFORMAL_COVERAGE:
        if t.can_emit(f"{window}:conformal_coverage"):
            logger.warning(
                "Conformal coverage below guarantee",
                extra={
                    "window": window,
                    "conformal_coverage": coverage,
                    "threshold": MIN_CONFORMAL_COVERAGE,
                    "severity": "warning",
                },
            )

    # ── Data quality alert ─────────────────────────────────────────────────
    # Low data quality means PSI, F1, and trust score are all unreliable -
    # alert before bad data corrupts any downstream monitoring decision.
    dq = summary.get("avg_data_quality_score")
    if dq is not None and dq < MIN_DATA_QUALITY_SCORE:
        if t.can_emit(f"{window}:data_quality"):
            logger.warning(
                "Data quality score below threshold - metrics may be unreliable",
                extra={
                    "window": window,
                    "data_quality_score": dq,
                    "threshold": MIN_DATA_QUALITY_SCORE,
                    "severity": "warning",
                },
            )

    # ── Output drift alert ─────────────────────────────────────────────────
    # Output drift often precedes input drift in detection time.  Alert when
    # the mean output PSI exceeds the warning threshold.
    od = summary.get("avg_output_drift_score")
    if od is not None and od > MAX_OUTPUT_DRIFT_SCORE:
        if t.can_emit(f"{window}:output_drift"):
            logger.warning(
                "Output distribution drift exceeds warning threshold",
                extra={
                    "window": window,
                    "output_drift_score": od,
                    "threshold": MAX_OUTPUT_DRIFT_SCORE,
                    "severity": "warning",
                },
            )

    # ── MMD joint-distribution drift ─────────────────────────────────────────
    # PSI checks each feature's marginal distribution independently.  MMD tests
    # the full joint distribution - it fires when correlation structure shifts
    # while all marginals look stable.  This is the highest-value alert in the
    # system: it catches drift that univariate monitoring misses entirely.
    mmd_p = summary.get("mmd_p_value")
    mmd_drift = summary.get("mmd_is_drift")
    if mmd_drift and mmd_p is not None:
        if t.can_emit(f"{window}:mmd_drift"):
            logger.warning(
                "Joint distribution shift detected by MMD",
                extra={
                    "window": window,
                    "mmd_p_value": mmd_p,
                    "psi": summary.get("avg_drift_score"),
                    "severity": "warning",
                    "note": (
                        "MMD detected joint drift not visible to per-feature PSI. "
                        "Check feature correlations and interaction structure."
                    ),
                },
            )


# ---------------------------------------------------------------------------
# Webhook alerting
# ---------------------------------------------------------------------------

# Type alias for the injectable HTTP POST callable.
# Matches the signature of ``requests.post`` for the fields we use.
_PostFn = Callable[..., Any]


class WebhookAlerter:
    """
    Fires HTTP POST notifications when trust score crosses a threshold.

    Designed to integrate with Slack incoming webhooks, PagerDuty Events
    API v2, or any HTTP endpoint that accepts a JSON body.  Network errors
    are caught and logged rather than raised so a misconfigured or
    unavailable webhook never interrupts the monitoring pipeline.

    The ``_post`` parameter exists purely for testing: pass a mock or stub
    in tests so no real HTTP calls are made.  The default is
    ``requests.post``, loaded lazily to avoid adding a hard import-time
    dependency on ``requests`` in environments that do not use webhooks.

    Example - Slack webhook::

        alerter = WebhookAlerter(
            url="https://hooks.slack.com/services/T.../B.../...",
            severity_filter="critical",   # only page on critical
        )
        # call alerter.notify(...) from check_alerts or the aggregation loop

    Example - custom endpoint::

        alerter = WebhookAlerter(url="https://alerts.internal/model-monitor")

    Args:
        url:             Webhook endpoint URL.
        timeout_s:       HTTP request timeout in seconds.  Defaults to 2.0
                         so a slow endpoint cannot stall the aggregation loop.
        severity_filter: When set, only fire for alerts at this severity or
                         above (``"warning"`` fires on both warning and
                         critical; ``"critical"`` fires only on critical).
                         ``None`` (default) fires on all severities.
        _post:           Injectable HTTP POST callable.  Defaults to
                         ``requests.post``.  Override in tests only.
    """

    _SEVERITY_RANK: dict[str, int] = {"warning": 1, "critical": 2}

    def __init__(
        self,
        *,
        url: str,
        timeout_s: float = 2.0,
        severity_filter: str | None = None,
        _post: _PostFn | None = None,
    ) -> None:
        if not url:
            raise ValueError("WebhookAlerter requires a non-empty url")
        if severity_filter is not None and severity_filter not in self._SEVERITY_RANK:
            raise ValueError(
                f"severity_filter must be one of {list(self._SEVERITY_RANK)}, "
                f"got {severity_filter!r}"
            )
        self._url = url
        self._timeout_s = timeout_s
        self._severity_filter = severity_filter
        self._post = _post  # None → resolved lazily to requests.post

    def _resolve_post(self) -> _PostFn:
        if self._post is not None:
            return self._post
        if not _REQUESTS_AVAILABLE:
            raise RuntimeError(
                "requests is required for webhook alerts. "
                "Install with: pip install requests"
            )
        return _requests_lib.post

    def notify(
        self,
        *,
        window: str,
        trust_score: float,
        severity: str,
        context: dict[str, object] | None = None,
    ) -> None:
        """
        POST an alert payload to the configured webhook URL.

        The call is best-effort: any exception (network error, timeout,
        non-2xx response) is logged at WARNING level and swallowed so
        webhook failures never block the monitoring pipeline.

        Payload schema::

            {
                "window":              "5m",
                "trust_score":         0.52,
                "severity":            "critical",
                "ts":                  1713967200.0,
                "drift_score":         0.18,
                "output_drift_score":  0.12,
                "data_quality_score":  0.94,
                "conformal_coverage":  0.88,
                "f1":                  0.79,
                "runbook_url":         "https://..."  // if MODEL_MONITOR_RUNBOOK_URL set
            }

        Args:
            window:      aggregation window label (e.g. ``"5m"``).
            trust_score: current trust score that triggered the alert.
            severity:    ``"warning"`` or ``"critical"``.
        """
        if severity not in self._SEVERITY_RANK:
            raise ValueError(
                f"severity must be one of {list(self._SEVERITY_RANK)}, got {severity!r}"
            )

        # Apply severity filter - skip if this alert is below the threshold.
        if self._severity_filter is not None:
            if (
                self._SEVERITY_RANK[severity]
                < self._SEVERITY_RANK[self._severity_filter]
            ):
                return

        # Rich payload for PagerDuty / Opsgenie / Slack webhook routing.
        # Include full monitoring context so operators can triage without
        # opening Grafana. Callers pass context=summary_fields to populate.
        import os as _os

        payload: dict[str, Any] = {
            "window": window,
            "trust_score": round(trust_score, 4),
            "severity": severity,
            "ts": time.time(),
        }
        if context:
            for key in (
                "avg_drift_score",
                "avg_output_drift_score",
                "avg_data_quality_score",
                "avg_conformal_coverage",
                "avg_conformal_set_size",
                "avg_calibration_error",
                "avg_f1",
                "n_batches",
            ):
                if key in context:
                    payload[key.replace("avg_", "")] = context[key]
        runbook = _os.environ.get("MODEL_MONITOR_RUNBOOK_URL")
        if runbook:
            payload["runbook_url"] = runbook

        post = self._resolve_post()
        try:
            resp = post(self._url, json=payload, timeout=self._timeout_s)
            resp.raise_for_status()
        except Exception as exc:
            # A failed webhook must never crash or block the pipeline.
            logger.warning(
                "webhook_alert_failed",
                extra={"url": self._url, "error": str(exc)},
            )
