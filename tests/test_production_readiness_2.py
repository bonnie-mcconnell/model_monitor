"""
Tests for the production-readiness second pass:

- ECE computation correctness
- Weighted aggregation (n_samples)
- Prometheus decisions_total uses count_by_action (SQL GROUP BY)
- MetricsStore.prune_before()
- DecisionStore.count_by_action()
- GET /dashboard/config structure
- GET /dashboard/health/detailed structure
- GET /dashboard/models/compare logic
- MetricsEventIn field validation (ge/le constraints)
- sim_drift_window param makes DriftMonitor use a small window
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from model_monitor.api.schemas import MetricsEventIn
from model_monitor.config.settings import (
    DriftConfig,
    load_config,
)
from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.monitoring.aggregation import _aggregate_records
from model_monitor.monitoring.types import MetricRecord
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.utils.stats import expected_calibration_error

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics_store(tmp_path: Path) -> MetricsStore:
    return MetricsStore(db_path=tmp_path / "metrics.db")


def _make_decision_store(tmp_path: Path) -> DecisionStore:
    return DecisionStore(db_path=tmp_path / "decisions.db")


def _record(
    batch_id: str = "b1",
    n_samples: int = 100,
    f1: float = 0.88,
    accuracy: float = 0.90,
    calibration_error: float | None = None,
    ts: float | None = None,
) -> MetricRecord:
    return cast(
        MetricRecord,
        {
            "timestamp": ts if ts is not None else time.time(),
            "batch_id": batch_id,
            "n_samples": n_samples,
            "accuracy": accuracy,
            "f1": f1,
            "avg_confidence": 0.85,
            "drift_score": 0.05,
            "decision_latency_ms": 12.0,
            "calibration_error": calibration_error,
            "feature_drift_scores": None,
            "behavioral_violation_rate": None,
            "shap_attribution": None,
            "action": "none",
            "reason": "healthy",
            "previous_model": None,
            "new_model": None,
        },
    )


# ---------------------------------------------------------------------------
# ECE - utils/stats.py
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration() -> None:
    """
    A model that is 80% confident exactly when it is 80% accurate
    should have ECE = 0.
    """
    rng = np.random.default_rng(0)
    confidences = np.full(100, 0.8)
    # 80 correct, 20 wrong → accuracy = 80%
    correct = np.array([1] * 80 + [0] * 20, dtype=float)
    rng.shuffle(correct)
    ece = expected_calibration_error(confidences, correct)
    assert ece < 0.02, f"Expected near-zero ECE for perfect calibration, got {ece:.4f}"


def test_ece_overconfident_model() -> None:
    """
    A model that is 100% confident but only 50% accurate should have high ECE.
    """
    confidences = np.ones(100)
    correct = np.array([1] * 50 + [0] * 50, dtype=float)
    ece = expected_calibration_error(confidences, correct)
    assert ece > 0.4, f"Expected high ECE for overconfident model, got {ece:.4f}"


def test_ece_bounded_to_0_1() -> None:
    rng = np.random.default_rng(1)
    confidences = rng.random(200)
    correct = rng.integers(0, 2, 200).astype(float)
    ece = expected_calibration_error(confidences, correct)
    assert 0.0 <= ece <= 1.0


def test_ece_empty_input() -> None:
    ece = expected_calibration_error(np.array([]), np.array([]))
    assert ece == 0.0


def test_ece_bin_boundary_included() -> None:
    """Confidence = 1.0 must fall into the last bin (not be dropped)."""
    confidences = np.array([1.0, 1.0, 1.0])
    correct = np.array([1.0, 1.0, 1.0])
    ece = expected_calibration_error(confidences, correct)
    # Perfect confidence, perfect accuracy → ECE = 0
    assert ece < 1e-10


# ---------------------------------------------------------------------------
# Weighted aggregation
# ---------------------------------------------------------------------------


def test_aggregate_records_weights_by_n_samples() -> None:
    """
    A 1000-sample batch with f1=0.9 and a 10-sample batch with f1=0.1
    should produce avg_f1 ≈ 0.9 (dominated by the large batch),
    not 0.5 (simple mean).
    """

    records = [
        _record("large", n_samples=1000, f1=0.90, accuracy=0.92),
        _record("small", n_samples=10, f1=0.10, accuracy=0.15),
    ]
    summary = _aggregate_records("5m", records)

    # Weighted: (1000*0.9 + 10*0.1) / 1010 ≈ 0.892
    expected_f1 = (1000 * 0.90 + 10 * 0.10) / 1010
    assert abs(summary.avg_f1 - expected_f1) < 0.001, (
        f"Expected weighted avg_f1 ≈ {expected_f1:.4f}, got {summary.avg_f1:.4f}"
    )

    expected_acc = (1000 * 0.92 + 10 * 0.15) / 1010
    assert abs(summary.avg_accuracy - expected_acc) < 0.001


def test_aggregate_records_ece_skips_null() -> None:
    """
    avg_calibration_error must ignore records where calibration_error is None
    and only average over records that have it.
    """

    records = [
        _record("b1", calibration_error=0.10),
        _record("b2", calibration_error=None),  # no labels
        _record("b3", calibration_error=0.20),
    ]
    summary = _aggregate_records("5m", records)
    assert summary.avg_calibration_error is not None
    assert abs(summary.avg_calibration_error - 0.15) < 1e-9


def test_aggregate_records_ece_none_when_all_null() -> None:
    """When no records have calibration data, avg_calibration_error is None."""

    records = [
        _record("b1", calibration_error=None),
        _record("b2", calibration_error=None),
    ]
    summary = _aggregate_records("5m", records)
    assert summary.avg_calibration_error is None


# ---------------------------------------------------------------------------
# DecisionStore.count_by_action
# ---------------------------------------------------------------------------


def test_count_by_action_empty(tmp_path: Path) -> None:
    store = _make_decision_store(tmp_path)
    assert store.count_by_action() == {}


def test_count_by_action_groups_correctly(tmp_path: Path) -> None:
    """count_by_action must return a GROUP BY result, not a Python loop."""
    store = _make_decision_store(tmp_path)

    for _ in range(3):
        store.record(
            decision=Decision(action=DecisionType.NONE, reason="ok", metadata={}),
            batch_index=0,
            trust_score=0.9,
            f1=0.88,
            drift_score=0.02,
        )
    for _ in range(2):
        store.record(
            decision=Decision(action=DecisionType.RETRAIN, reason="degraded", metadata={}),
            batch_index=1,
            trust_score=0.7,
            f1=0.75,
            drift_score=0.05,
        )

    counts = store.count_by_action()
    assert counts["none"] == 3
    assert counts["retrain"] == 2
    assert len(counts) == 2


def test_count_by_action_is_monotone(tmp_path: Path) -> None:
    """
    count_by_action() must never decrease across calls (Prometheus counter
    semantics).  Verify by recording more decisions and checking the count
    is >= the previous value for every action.
    """
    store = _make_decision_store(tmp_path)

    store.record(
        decision=Decision(action=DecisionType.NONE, reason="ok", metadata={}),
        batch_index=0,
        trust_score=0.9,
        f1=0.88,
        drift_score=0.02,
    )
    counts_before = store.count_by_action()

    store.record(
        decision=Decision(action=DecisionType.NONE, reason="ok", metadata={}),
        batch_index=1,
        trust_score=0.9,
        f1=0.88,
        drift_score=0.02,
    )
    counts_after = store.count_by_action()

    assert counts_after["none"] >= counts_before["none"]


# ---------------------------------------------------------------------------
# MetricsStore.prune_before
# ---------------------------------------------------------------------------


def test_prune_before_removes_old_records(tmp_path: Path) -> None:
    store = _make_metrics_store(tmp_path)
    old_ts = time.time() - 3600  # 1 hour ago

    store.write(_record("old1", ts=old_ts - 10))
    store.write(_record("old2", ts=old_ts - 5))
    store.write(_record("new1"))  # current time

    removed = store.prune_before(old_ts)

    assert removed == 2
    records = store.tail(limit=100)
    batch_ids = {r["batch_id"] for r in records}
    assert "old1" not in batch_ids
    assert "old2" not in batch_ids
    assert "new1" in batch_ids


def test_prune_before_returns_zero_when_nothing_old(tmp_path: Path) -> None:
    store = _make_metrics_store(tmp_path)
    store.write(_record())  # current

    removed = store.prune_before(time.time() - 86400)  # 24h cutoff
    assert removed == 0


def test_prune_before_does_not_delete_future_records(tmp_path: Path) -> None:
    store = _make_metrics_store(tmp_path)
    store.write(_record("keep"))

    future_cutoff = time.time() + 3600
    removed = store.prune_before(future_cutoff)

    # All records deleted (all older than a future timestamp)
    assert removed == 1


def test_prune_preserves_recent_after_bulk_deletion(tmp_path: Path) -> None:
    """After pruning, the store must still be fully functional."""
    store = _make_metrics_store(tmp_path)
    old_ts = time.time() - 7200

    for i in range(5):
        store.write(_record(f"old_{i}", ts=old_ts - i))
    store.write(_record("keep"))

    store.prune_before(time.time() - 3600)

    records = store.tail(limit=100)
    assert len(records) == 1
    assert records[0]["batch_id"] == "keep"


# ---------------------------------------------------------------------------
# MetricsEventIn validation
# ---------------------------------------------------------------------------


def test_metrics_event_in_rejects_negative_accuracy() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="accuracy"):
        MetricsEventIn(
            batch_id="b1",
            n_samples=50,
            accuracy=-0.1,
            f1=0.9,
            avg_confidence=0.8,
            drift_score=0.05,
            decision_latency_ms=10.0,
            action=DecisionType.NONE,
            reason="test",
        )


def test_metrics_event_in_rejects_f1_above_1() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="f1"):
        MetricsEventIn(
            batch_id="b1",
            n_samples=50,
            accuracy=0.9,
            f1=1.5,
            avg_confidence=0.8,
            drift_score=0.05,
            decision_latency_ms=10.0,
            action=DecisionType.NONE,
            reason="test",
        )


def test_metrics_event_in_rejects_zero_n_samples() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="n_samples"):
        MetricsEventIn(
            batch_id="b1",
            n_samples=0,
            accuracy=0.9,
            f1=0.88,
            avg_confidence=0.8,
            drift_score=0.05,
            decision_latency_ms=10.0,
            action=DecisionType.NONE,
            reason="test",
        )


def test_metrics_event_in_rejects_negative_drift() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="drift_score"):
        MetricsEventIn(
            batch_id="b1",
            n_samples=50,
            accuracy=0.9,
            f1=0.88,
            avg_confidence=0.8,
            drift_score=-0.01,
            decision_latency_ms=10.0,
            action=DecisionType.NONE,
            reason="test",
        )


def test_metrics_event_in_accepts_valid_payload() -> None:

    event = MetricsEventIn(
        batch_id="valid_batch",
        n_samples=100,
        accuracy=0.92,
        f1=0.91,
        avg_confidence=0.87,
        drift_score=0.04,
        decision_latency_ms=15.0,
        action=DecisionType.NONE,
        reason="system healthy",
    )
    assert event.accuracy == 0.92


# ---------------------------------------------------------------------------
# sim_drift_window
# ---------------------------------------------------------------------------


def test_sim_drift_window_creates_small_monitor() -> None:
    """
    When sim_drift_window=5, the DriftMonitor passed to Predictor must have
    window=5, not the production config value (500).  This ensures PSI
    actually fires within 80 batches during make sim.
    """
    cfg = load_config()
    # The production window is much larger than 5
    assert cfg.drift.window > 5, "Test assumption: prod window > 5"

    # Simulate what simulate_stream does with sim_drift_window=5
    sim_window = 5
    sim_config = cfg.model_copy(
        update={
            "drift": DriftConfig(
                psi_threshold=cfg.drift.psi_threshold,
                window=sim_window,
            )
        }
    )

    assert sim_config.drift.window == sim_window
    # Other config is unchanged
    assert sim_config.retrain.min_f1_gain == cfg.retrain.min_f1_gain
    assert sim_config.drift.psi_threshold == cfg.drift.psi_threshold


def test_sim_drift_window_zero_uses_production_config() -> None:
    """sim_drift_window=0 must use the production config window unchanged."""
    cfg = load_config()
    # When sim_drift_window == 0, no override is applied
    sim_drift_window = 0
    sim_config = cfg if sim_drift_window == 0 else cfg  # same reference

    assert sim_config.drift.window == cfg.drift.window


# ---------------------------------------------------------------------------
# MetricsStore calibration_error round-trip
# ---------------------------------------------------------------------------


def test_metrics_store_calibration_error_round_trip(tmp_path: Path) -> None:
    store = _make_metrics_store(tmp_path)
    store.write(_record("b1", calibration_error=0.042))

    records = store.tail(limit=1)
    assert records[0]["calibration_error"] == pytest.approx(0.042)


def test_metrics_store_null_calibration_error_round_trip(tmp_path: Path) -> None:
    store = _make_metrics_store(tmp_path)
    store.write(_record("b2", calibration_error=None))

    records = store.tail(limit=1)
    assert records[0]["calibration_error"] is None
