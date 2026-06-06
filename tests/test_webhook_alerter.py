"""
Tests for monitoring/alerting.WebhookAlerter.

WebhookAlerter is the outbound notification path that turns trust-score
events into real external alerts (Slack, PagerDuty, etc.).  The key
properties under test:

1. Correct payload shape - downstream consumers depend on the schema.
2. Timeout is forwarded - a slow endpoint must not stall the pipeline.
3. Network errors are swallowed - a misconfigured webhook must never
   crash the aggregation loop.
4. severity_filter suppresses lower-priority alerts - avoids noise when
   only critical pages are wanted.
5. Invalid constructor arguments raise immediately - fail-fast at
   configuration time, not at the first alert.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from model_monitor.monitoring.alerting import WebhookAlerter

_URL = "https://hooks.example.com/test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_post(status_code: int = 200) -> MagicMock:
    """Return a mock that behaves like a successful requests.post call."""
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    post = MagicMock(return_value=response)
    return post


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_empty_url_raises() -> None:
    with pytest.raises(ValueError, match="non-empty url"):
        WebhookAlerter(url="")


def test_invalid_severity_filter_raises() -> None:
    with pytest.raises(ValueError, match="severity_filter"):
        WebhookAlerter(url=_URL, severity_filter="info")


def test_valid_construction_does_not_raise() -> None:
    alerter = WebhookAlerter(url=_URL)
    assert alerter is not None


# ---------------------------------------------------------------------------
# Payload correctness
# ---------------------------------------------------------------------------


def test_notify_posts_correct_payload() -> None:
    """
    Payload must include window, trust_score, severity, and a numeric ts.
    Downstream consumers (Slack, PagerDuty adapters) depend on this schema.
    """
    post = _make_mock_post()
    alerter = WebhookAlerter(url=_URL, _post=post)

    alerter.notify(window="5m", trust_score=0.52, severity="critical")

    post.assert_called_once()
    _, kwargs = post.call_args
    payload = kwargs["json"]

    assert payload["window"] == "5m"
    assert payload["trust_score"] == 0.52
    assert payload["severity"] == "critical"
    assert isinstance(payload["ts"], float)


def test_notify_posts_to_correct_url() -> None:
    post = _make_mock_post()
    alerter = WebhookAlerter(url=_URL, _post=post)
    alerter.notify(window="1h", trust_score=0.65, severity="warning")

    args, _ = post.call_args
    assert args[0] == _URL


def test_notify_forwards_timeout() -> None:
    """
    The configured timeout must be passed to the POST call so a slow
    endpoint cannot block the aggregation loop.
    """
    post = _make_mock_post()
    alerter = WebhookAlerter(url=_URL, timeout_s=1.5, _post=post)
    alerter.notify(window="5m", trust_score=0.55, severity="warning")

    _, kwargs = post.call_args
    assert kwargs["timeout"] == 1.5


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_notify_does_not_raise_on_connection_error() -> None:
    """
    A broken webhook must never crash the monitoring pipeline.
    The error is logged but swallowed.
    """

    def failing_post(*args: object, **kwargs: object) -> None:
        raise ConnectionError("unreachable")

    alerter = WebhookAlerter(url=_URL, _post=failing_post)

    # Must not raise
    alerter.notify(window="5m", trust_score=0.40, severity="critical")


def test_notify_does_not_raise_on_http_error() -> None:
    """
    A non-2xx response triggers raise_for_status which raises an
    HTTPError.  WebhookAlerter must catch it.
    """
    import requests

    response = MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    post = MagicMock(return_value=response)

    alerter = WebhookAlerter(url=_URL, _post=post)
    alerter.notify(window="5m", trust_score=0.40, severity="critical")  # no raise


def test_connection_error_logged_as_warning(caplog: pytest.LogCaptureFixture) -> None:
    """
    Failed webhook calls are logged at WARNING so operators can detect
    misconfigured URLs in structured logs without crashing the pipeline.
    """
    import logging

    def failing_post(*args: object, **kwargs: object) -> None:
        raise ConnectionError("host unreachable")

    alerter = WebhookAlerter(url=_URL, _post=failing_post)

    with caplog.at_level(logging.WARNING, logger="model_monitor.alerts"):
        alerter.notify(window="5m", trust_score=0.40, severity="critical")

    assert any("webhook_alert_failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Severity filter
# ---------------------------------------------------------------------------


def test_severity_filter_critical_suppresses_warning() -> None:
    """
    When severity_filter="critical", warning-level alerts must be dropped
    so on-call engineers only receive actionable pages.
    """
    post = _make_mock_post()
    alerter = WebhookAlerter(url=_URL, severity_filter="critical", _post=post)

    alerter.notify(window="5m", trust_score=0.68, severity="warning")

    post.assert_not_called()


def test_severity_filter_critical_allows_critical() -> None:
    post = _make_mock_post()
    alerter = WebhookAlerter(url=_URL, severity_filter="critical", _post=post)

    alerter.notify(window="5m", trust_score=0.45, severity="critical")

    post.assert_called_once()


def test_severity_filter_warning_allows_both() -> None:
    """severity_filter="warning" must pass through warning AND critical."""
    post = _make_mock_post()
    alerter = WebhookAlerter(url=_URL, severity_filter="warning", _post=post)

    alerter.notify(window="5m", trust_score=0.68, severity="warning")
    alerter.notify(window="5m", trust_score=0.45, severity="critical")

    assert post.call_count == 2


def test_no_severity_filter_allows_all() -> None:
    post = _make_mock_post()
    alerter = WebhookAlerter(url=_URL, _post=post)

    alerter.notify(window="5m", trust_score=0.68, severity="warning")
    alerter.notify(window="5m", trust_score=0.45, severity="critical")

    assert post.call_count == 2


# ---------------------------------------------------------------------------
# Invalid severity in notify call
# ---------------------------------------------------------------------------


def test_notify_raises_on_invalid_severity() -> None:
    """
    An unrecognised severity string is a programming error - fail loudly
    rather than silently sending a malformed payload.
    """
    post = _make_mock_post()
    alerter = WebhookAlerter(url=_URL, _post=post)

    with pytest.raises(ValueError, match="severity"):
        alerter.notify(window="5m", trust_score=0.5, severity="info")


# ---------------------------------------------------------------------------
# Lazy import of requests (no import-time side effects)
# ---------------------------------------------------------------------------


def test_webhook_alerter_importable_without_requests() -> None:
    """
    WebhookAlerter must be constructible without importing requests at
    module load time.  requests is loaded lazily only when notify() is
    called without an injected _post.
    """
    with patch.dict("sys.modules", {"requests": None}):
        # Construction must succeed - requests not imported yet
        alerter = WebhookAlerter(url=_URL)
        assert alerter is not None
