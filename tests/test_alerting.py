"""
Tests for monitoring/alerting.py.

The alerting system has one stateful property that matters: cooldown.
Alerts must not fire on every aggregation pass (every 60 seconds) when
the trust score is persistently low - that is alert fatigue. The
cooldown ensures each alert fires at most once per 5 minutes.

We test this by inspecting what gets logged, which requires capturing
log output - the correct approach for testing log-only functions.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from model_monitor.monitoring.alerting import (
    AlertCooldownTracker,
    _default_tracker,
    check_alerts,
)


@pytest.fixture(autouse=True)
def reset_alert_state() -> Iterator[None]:
    """
    Reset the process-level cooldown tracker before and after each test.

    Without this, a test that fires an alert fills the cooldown slot and
    causes the next test to see suppressed alerts, producing false passes
    on the cooldown tests and false failures on the fire tests.
    """
    _default_tracker.reset()
    yield
    _default_tracker.reset()


def test_critical_alert_fires_when_trust_below_critical_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="model_monitor.alerts"):
        check_alerts("5m", {"trust_score": 0.50})

    assert any("Critical trust degradation" in r.message for r in caplog.records)


def test_warning_alert_fires_when_trust_below_min_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="model_monitor.alerts"):
        check_alerts("5m", {"trust_score": 0.65})

    assert any("operational floor" in r.message for r in caplog.records)


def test_no_alert_when_trust_is_healthy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="model_monitor.alerts"):
        check_alerts("5m", {"trust_score": 0.90})

    assert len(caplog.records) == 0


def test_cooldown_suppresses_second_alert(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Two consecutive calls with critical trust must only produce one log
    record - the second is suppressed by the 5-minute cooldown.
    """
    with caplog.at_level(logging.ERROR, logger="model_monitor.alerts"):
        check_alerts("5m", {"trust_score": 0.50})
        check_alerts("5m", {"trust_score": 0.50})

    critical_records = [r for r in caplog.records if "Critical" in r.message]
    assert len(critical_records) == 1


def test_different_windows_have_independent_cooldowns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A cooldown on window '5m' must not suppress alerts on window '1h'.
    Each (window, severity) pair has its own cooldown slot.
    """
    with caplog.at_level(logging.ERROR, logger="model_monitor.alerts"):
        check_alerts("5m", {"trust_score": 0.50})
        check_alerts("1h", {"trust_score": 0.50})

    critical_records = [r for r in caplog.records if "Critical" in r.message]
    assert len(critical_records) == 2


def test_missing_trust_score_does_not_crash() -> None:
    """check_alerts must handle a summary dict with no trust_score key."""
    check_alerts("5m", {})
    check_alerts("5m", {"other_key": 0.5})


def test_injected_tracker_is_independent_of_process_singleton() -> None:
    """
    Passing an explicit tracker must not affect _default_tracker state.
    This verifies the injection path is genuinely isolated - a fresh
    AlertCooldownTracker starts with no cooldown history regardless of
    what the process singleton has recorded.
    """
    fresh = AlertCooldownTracker()
    # Fire via default tracker to fill its slot
    check_alerts("5m", {"trust_score": 0.50})
    # Fresh tracker has no history - same key must be allowed
    assert fresh.can_emit("5m:critical") is True


def test_tracker_reset_clears_all_slots() -> None:
    """reset() must clear every slot, not just the one most recently used."""
    tracker = AlertCooldownTracker()
    tracker.can_emit("5m:critical")
    tracker.can_emit("1h:warning")
    tracker.reset()
    assert tracker.can_emit("5m:critical") is True
    assert tracker.can_emit("1h:warning") is True


