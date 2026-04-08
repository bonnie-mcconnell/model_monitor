"""
Tests verifying that behavioral_violation_rate flows from BehavioralDecisionStore
through aggregate_once into compute_trust_score.

This is the integration test for the closed loop between the two monitoring
systems. A model failing behavioral contracts must produce a lower trust score
than an identical model with no violations.
"""
from __future__ import annotations

import time
import uuid
from typing import cast

import pytest

from model_monitor.monitoring.aggregation import _aggregate_records
from model_monitor.monitoring.types import MetricRecord

# ---------------------------------------------------------------------------
# _aggregate_records - unit-level verification
# ---------------------------------------------------------------------------

def _make_metric_record(**overrides: object) -> MetricRecord:
    base = {
        "timestamp": time.time(),
        "batch_id": str(uuid.uuid4()),
        "n_samples": 100,
        "accuracy": 0.90,
        "f1": 0.88,
        "avg_confidence": 0.85,
        "drift_score": 0.02,
        "decision_latency_ms": 150.0,
        "action": "none",
        "reason": "within thresholds",
        "previous_model": None,
        "new_model": None,
    }
    return cast(MetricRecord, {**base, **overrides})


def test_aggregate_records_without_behavioral_rate_uses_zero() -> None:
    """
    When no behavioral_violation_rate is passed, it defaults to 0.0.
    The behavioral component in TrustScoreComponents should be 1.0
    (zero violations → full score).
    """
    records = [_make_metric_record()]
    summary = _aggregate_records("5m", records)

    assert summary.behavioral_violation_rate == 0.0
    assert summary.trust_components["behavioral"] == pytest.approx(1.0)


def test_aggregate_records_with_full_violation_rate_lowers_trust() -> None:
    """
    100% violation rate must produce a lower trust score than 0%.
    This is the core property of the behavioral wiring.
    """
    records = [_make_metric_record()]

    summary_clean = _aggregate_records("5m", records, behavioral_violation_rate=0.0)
    summary_violated = _aggregate_records("5m", records, behavioral_violation_rate=1.0)

    assert summary_violated.trust_score < summary_clean.trust_score


def test_aggregate_records_behavioral_rate_stored_in_summary() -> None:
    """
    The computed behavioral_violation_rate must be stored in AggregatedSummary
    so callers can inspect and log it.
    """
    records = [_make_metric_record()]
    summary = _aggregate_records("5m", records, behavioral_violation_rate=0.4)

    assert summary.behavioral_violation_rate == pytest.approx(0.4)
    assert summary.trust_components["behavioral"] == pytest.approx(0.6)


def test_aggregate_records_trust_score_bounded_with_any_violation_rate() -> None:
    """Trust score must stay in [0, 1] for any violation rate."""
    records = [_make_metric_record()]
    for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        summary = _aggregate_records("5m", records, behavioral_violation_rate=rate)
        assert 0.0 <= summary.trust_score <= 1.0
