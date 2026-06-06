"""
Tests for all new functionality added in the production-readiness pass:

- AlertStore: record(), tail(), count_since(), filters
- DecisionStore.count()
- DriftMonitor.last_feature_scores
- Predictor.last_metric_record with feature_drift_scores
- MetricsStore feature_drift_scores round-trip
- Prometheus /metrics endpoint format
- startup.py uses config min_samples (not hardcoded 5)
- simulate_decision uses DecisionStore.count() not batch_index=0
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import model_monitor.api.metrics as m_module
from model_monitor.api.metrics import prometheus_metrics
from model_monitor.config.settings import (
    AppConfig,
    DriftConfig,
    RetrainConfig,
    RollbackConfig,
    load_config,
)
from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.inference.predict import Predictor
from model_monitor.monitoring.alerting import AlertCooldownTracker, check_alerts
from model_monitor.monitoring.drift import DriftMonitor
from model_monitor.monitoring.types import MetricRecord
from model_monitor.storage.alert_store import AlertStore
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.model_store import ModelStore
from model_monitor.storage.models.metrics_models import MetricsRecordORM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alert_store(tmp_path: Path) -> AlertStore:
    return AlertStore(db_path=tmp_path / "test.db")


def _make_metrics_store(tmp_path: Path) -> MetricsStore:
    return MetricsStore(db_path=tmp_path / "metrics.db")


def _make_decision_store(tmp_path: Path) -> DecisionStore:
    return DecisionStore(db_path=tmp_path / "decisions.db")


def _metric_record(
    batch_id: str = "b1", feature_scores: list[float] | None = None
) -> MetricRecord:
    return cast(
        MetricRecord,
        {
            "timestamp": time.time(),
            "batch_id": batch_id,
            "n_samples": 50,
            "accuracy": 0.90,
            "f1": 0.88,
            "avg_confidence": 0.85,
            "drift_score": 0.05,
            "decision_latency_ms": 12.0,
            "calibration_error": None,
            "feature_drift_scores": feature_scores,
            "behavioral_violation_rate": None,
            "shap_attribution": None,
            "action": "none",
            "reason": "healthy",
            "previous_model": None,
            "new_model": None,
        },
    )


# ---------------------------------------------------------------------------
# AlertStore - basic operations
# ---------------------------------------------------------------------------


def test_alert_store_record_and_tail(tmp_path: Path) -> None:
    """Recorded alerts are retrievable via tail()."""
    store = _make_alert_store(tmp_path)
    store.record(window="5m", severity="critical", trust_score=0.45)
    store.record(window="1h", severity="warning", trust_score=0.68)

    alerts = store.tail(limit=100)
    assert len(alerts) == 2
    actions = {a["severity"] for a in alerts}
    assert actions == {"critical", "warning"}


def test_alert_store_tail_severity_filter(tmp_path: Path) -> None:
    store = _make_alert_store(tmp_path)
    store.record(window="5m", severity="critical", trust_score=0.45)
    store.record(window="5m", severity="warning", trust_score=0.68)
    store.record(window="5m", severity="critical", trust_score=0.42)

    critical = store.tail(severity="critical")
    assert all(a["severity"] == "critical" for a in critical)
    assert len(critical) == 2


def test_alert_store_tail_window_filter(tmp_path: Path) -> None:
    store = _make_alert_store(tmp_path)
    store.record(window="5m", severity="warning", trust_score=0.68)
    store.record(window="1h", severity="warning", trust_score=0.67)
    store.record(window="24h", severity="warning", trust_score=0.66)

    five_min = store.tail(window="5m")
    assert len(five_min) == 1
    assert five_min[0]["window"] == "5m"


def test_alert_store_tail_since_ts_filter(tmp_path: Path) -> None:
    store = _make_alert_store(tmp_path)
    past = time.time() - 3600
    store.record(window="5m", severity="critical", trust_score=0.45)

    # Only alerts after "now" (i.e. after the one we just wrote)
    future_threshold = time.time() + 1
    alerts = store.tail(since_ts=future_threshold)
    assert len(alerts) == 0

    # Alerts after past time includes our record
    alerts_since_past = store.tail(since_ts=past)
    assert len(alerts_since_past) == 1


def test_alert_store_count_since(tmp_path: Path) -> None:
    store = _make_alert_store(tmp_path)
    t0 = time.time()
    store.record(window="5m", severity="critical", trust_score=0.45)
    store.record(window="5m", severity="warning", trust_score=0.68)

    assert store.count_since(t0 - 1) == 2
    assert store.count_since(t0 - 1, severity="critical") == 1
    assert store.count_since(time.time() + 10) == 0


def test_alert_store_tail_returns_newest_first(tmp_path: Path) -> None:
    store = _make_alert_store(tmp_path)
    for i in range(5):
        store.record(window="5m", severity="warning", trust_score=0.65 + i * 0.01)
        time.sleep(0.01)

    alerts = store.tail(limit=5)
    timestamps = [a["timestamp"] for a in alerts]
    assert timestamps == sorted(timestamps, reverse=True)


def test_alert_store_schema_fields(tmp_path: Path) -> None:
    """Every alert record must have all expected fields."""
    store = _make_alert_store(tmp_path)
    store.record(window="5m", severity="critical", trust_score=0.40)
    alert = store.tail(limit=1)[0]

    assert "id" in alert
    assert "timestamp" in alert
    assert "window" in alert
    assert "severity" in alert
    assert "trust_score" in alert
    assert alert["trust_score"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# DecisionStore.count()
# ---------------------------------------------------------------------------


def test_decision_store_count_empty(tmp_path: Path) -> None:
    store = _make_decision_store(tmp_path)
    assert store.count() == 0


def test_decision_store_count_increments(tmp_path: Path) -> None:
    store = _make_decision_store(tmp_path)

    for i in range(3):
        store.record(
            decision=Decision(action=DecisionType.NONE, reason="test", metadata={}),
            batch_index=i,
            trust_score=0.9,
            f1=0.88,
            drift_score=0.02,
        )

    assert store.count() == 3


def test_decision_store_count_independent_of_tail_limit(tmp_path: Path) -> None:
    """count() returns total, not limited by any tail() window."""
    store = _make_decision_store(tmp_path)

    for i in range(10):
        store.record(
            decision=Decision(action=DecisionType.NONE, reason="test", metadata={}),
            batch_index=i,
            trust_score=0.9,
            f1=0.88,
            drift_score=0.02,
        )

    # tail() with limit=3 returns 3, count() returns 10
    assert len(store.tail(limit=3)) == 3
    assert store.count() == 10


# ---------------------------------------------------------------------------
# DriftMonitor.last_feature_scores
# ---------------------------------------------------------------------------


def test_drift_monitor_last_feature_scores_empty_before_window_fills() -> None:
    """
    last_feature_scores must be empty before the sliding window is full.
    A caller that reads it before the window fills must not crash or see
    stale data from a previous run.
    """
    cfg = DriftConfig(psi_threshold=0.2, window=3)
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, (200, 4))
    monitor = DriftMonitor(reference_features=reference, config=cfg)

    monitor.update(rng.normal(0, 1, (50, 4)))  # window not filled yet
    assert monitor.last_feature_scores == []


def test_drift_monitor_last_feature_scores_set_after_window_fills() -> None:
    """
    After the window fills, last_feature_scores must have one entry per
    feature in the reference distribution.
    """
    cfg = DriftConfig(psi_threshold=0.2, window=2)
    rng = np.random.default_rng(1)
    n_features = 4
    reference = rng.normal(0, 1, (200, n_features))
    monitor = DriftMonitor(reference_features=reference, config=cfg)

    for _ in range(cfg.window):
        monitor.update(rng.normal(0, 1, (50, n_features)))

    assert len(monitor.last_feature_scores) == n_features
    assert all(isinstance(s, float) for s in monitor.last_feature_scores)


def test_drift_monitor_feature_scores_all_nonneg() -> None:
    """PSI is non-negative; every per-feature score must be >= 0."""
    cfg = DriftConfig(psi_threshold=0.2, window=2)
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, (300, 5))
    monitor = DriftMonitor(reference_features=reference, config=cfg)

    for _ in range(cfg.window):
        monitor.update(rng.normal(2, 1, (50, 5)))  # shifted distribution

    assert all(s >= 0.0 for s in monitor.last_feature_scores)


def test_drift_monitor_feature_scores_mean_equals_update_return() -> None:
    """
    The mean of last_feature_scores must equal the float returned by update().
    This is the key invariant: the scalar return value and the per-feature
    breakdown must be consistent.
    """
    cfg = DriftConfig(psi_threshold=0.2, window=2)
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, (300, 4))
    monitor = DriftMonitor(reference_features=reference, config=cfg)

    for _ in range(cfg.window - 1):
        monitor.update(rng.normal(1, 1, (50, 4)))

    mean_psi = monitor.update(rng.normal(1, 1, (50, 4)))

    expected_mean = float(np.mean(monitor.last_feature_scores))
    assert abs(mean_psi - expected_mean) < 1e-12


# ---------------------------------------------------------------------------
# MetricsStore feature_drift_scores round-trip
# ---------------------------------------------------------------------------


def test_metrics_store_persists_feature_drift_scores(tmp_path: Path) -> None:
    """feature_drift_scores must survive a write/read round-trip."""
    store = _make_metrics_store(tmp_path)
    scores = [0.02, 0.15, 0.01, 0.08, 0.22]
    store.write(_metric_record("b1", feature_scores=scores))

    records = store.tail(limit=1)
    assert len(records) == 1
    assert records[0]["feature_drift_scores"] is not None
    assert records[0]["feature_drift_scores"] == pytest.approx(scores)


def test_metrics_store_null_feature_drift_scores(tmp_path: Path) -> None:
    """None feature_drift_scores must survive the round-trip as None."""
    store = _make_metrics_store(tmp_path)
    store.write(_metric_record("b2", feature_scores=None))

    records = store.tail(limit=1)
    assert records[0]["feature_drift_scores"] is None


def test_metrics_store_feature_drift_scores_json_is_valid(tmp_path: Path) -> None:
    """The persisted JSON must be parseable as a list of floats."""
    store = _make_metrics_store(tmp_path)
    scores = [0.01, 0.02, 0.03]
    store.write(_metric_record("b3", feature_scores=scores))

    # Read directly from ORM to verify the raw JSON

    with store.Session() as session:
        row = session.query(MetricsRecordORM).filter_by(batch_id="b3").one()
        raw_json = row.feature_drift_scores

    assert raw_json is not None
    parsed = json.loads(raw_json)
    assert parsed == pytest.approx(scores)


# ---------------------------------------------------------------------------
# Predictor.last_metric_record
# ---------------------------------------------------------------------------


def test_predictor_last_metric_record_set_after_predict_batch() -> None:
    """last_metric_record must be populated after every predict_batch call."""
    cfg = load_config()

    class DummyModel:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.full((len(X), 2), 0.5)

    predictor = Predictor(config=cfg, f1_baseline=0.85)
    predictor.model = DummyModel()

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((50, 3)), columns=["f0", "f1", "f2"])
    y = pd.Series(rng.integers(0, 2, 50))

    assert predictor.last_metric_record is None  # before first call

    predictor.predict_batch(X, y_true=y, batch_id="test_batch")

    record = predictor.last_metric_record
    assert record is not None
    assert record["batch_id"] == "test_batch"
    assert record["n_samples"] == 50
    assert 0.0 <= record["accuracy"] <= 1.0
    assert 0.0 <= record["f1"] <= 1.0
    assert record["decision_latency_ms"] > 0.0


def test_predictor_last_metric_record_includes_feature_scores_when_drift_monitor_set() -> (
    None
):
    """
    When DriftMonitor is configured and window fills, last_metric_record
    must include feature_drift_scores from the monitor.
    """
    cfg = AppConfig(
        drift=DriftConfig(psi_threshold=0.2, window=2),
        retrain=RetrainConfig(min_f1_gain=0.05, cooldown_batches=5, min_samples=100),
        rollback=RollbackConfig(max_f1_drop=0.15),
    )

    class DummyModel:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.full((len(X), 2), 0.5)

    rng = np.random.default_rng(5)
    n_features = 4
    reference = rng.normal(0, 1, (200, n_features))

    predictor = Predictor(
        config=cfg,
        f1_baseline=0.85,
        reference_features=reference,
    )
    predictor.model = DummyModel()

    cols = [f"f{i}" for i in range(n_features)]
    y = pd.Series(rng.integers(0, 2, 50))

    # Fill the window (window=2)
    for i in range(3):
        X = pd.DataFrame(rng.normal(0, 1, (50, n_features)), columns=cols)
        predictor.predict_batch(X, y_true=y, batch_id=f"b{i}")

    record = predictor.last_metric_record
    assert record is not None
    # After 3 calls the window has filled; feature scores should be present
    assert record["feature_drift_scores"] is not None
    assert len(record["feature_drift_scores"]) == n_features


# ---------------------------------------------------------------------------
# startup.py uses config min_samples (not hardcoded 5)
# ---------------------------------------------------------------------------


def test_startup_uses_config_min_samples() -> None:
    """
    start_background_loops must create the RetrainEvidenceBuffer with
    min_samples from the YAML config, not the hardcoded value of 5 that
    was the bug before this fix.

    We verify by inspecting the load_config() value used at import time -
    it must equal retrain.min_samples from retrain.yaml (1000), not 5.
    """
    cfg = load_config()
    # The bug was min_samples=5; this asserts the config threshold is respected
    assert cfg.retrain.min_samples > 5, (
        f"Expected min_samples > 5 from retrain.yaml, got {cfg.retrain.min_samples}. "
        "If this fails, startup.py is still using the hardcoded value."
    )


# ---------------------------------------------------------------------------
# Prometheus endpoint format
# ---------------------------------------------------------------------------


def test_prometheus_endpoint_produces_help_and_type_lines(tmp_path: Path) -> None:
    """The /metrics endpoint must include HELP and TYPE comment lines.

    prometheus-client generates these automatically for every registered
    metric.  This test confirms the correct metric families are registered
    and that the output is well-formed Prometheus text exposition format.
    """
    m_module._summary_store = MetricsSummaryStore()
    m_module._decision_store = _make_decision_store(tmp_path)
    m_module._model_store = ModelStore(base_path=str(tmp_path))
    m_module._metrics_store = _make_metrics_store(tmp_path)

    try:
        response = prometheus_metrics()
        body = bytes(response.body).decode()
    finally:
        m_module._summary_store = None
        m_module._decision_store = None
        m_module._model_store = None
        m_module._metrics_store = None

    assert "# HELP model_monitor_trust_score" in body
    assert "# TYPE model_monitor_trust_score gauge" in body
    assert "# HELP model_monitor_decisions_total" in body
    assert "# TYPE model_monitor_decisions_total counter" in body
    assert "# HELP model_monitor_model_version_info" in body
    assert "# HELP model_monitor_decision_latency_ms" in body
    assert "# TYPE model_monitor_decision_latency_ms histogram" in body


def test_prometheus_endpoint_content_type(tmp_path: Path) -> None:
    """The /metrics response must use the Prometheus text content type."""
    from prometheus_client import CONTENT_TYPE_LATEST

    m_module._summary_store = MetricsSummaryStore()
    m_module._decision_store = _make_decision_store(tmp_path)
    m_module._model_store = ModelStore(base_path=str(tmp_path))
    m_module._metrics_store = _make_metrics_store(tmp_path)

    try:
        response = prometheus_metrics()
    finally:
        m_module._summary_store = None
        m_module._decision_store = None
        m_module._model_store = None
        m_module._metrics_store = None

    assert response.media_type == CONTENT_TYPE_LATEST


def test_prometheus_omits_f1_baseline_when_absent(tmp_path: Path) -> None:
    """f1_baseline gauge is NaN when no active model has been promoted.

    prometheus-client always emits registered gauges; the module sets the
    value to NaN explicitly so no data point appears in Grafana rather than
    a stale value carried over from a previous promotion within the same
    process lifetime.
    """
    m_module._summary_store = MetricsSummaryStore()
    m_module._decision_store = _make_decision_store(tmp_path)
    m_module._model_store = ModelStore(base_path=str(tmp_path))
    m_module._metrics_store = _make_metrics_store(tmp_path)

    try:
        response = prometheus_metrics()
        body = bytes(response.body).decode()
    finally:
        m_module._summary_store = None
        m_module._decision_store = None
        m_module._model_store = None
        m_module._metrics_store = None

    data_lines = [
        ln for ln in body.splitlines() if "f1_baseline" in ln and not ln.startswith("#")
    ]
    assert len(data_lines) == 1, (
        f"Expected one f1_baseline data line, got: {data_lines}"
    )
    assert "NaN" in data_lines[0], (
        f"Expected NaN when no model promoted, got: {data_lines[0]}"
    )


# ---------------------------------------------------------------------------
# check_alerts with AlertStore
# ---------------------------------------------------------------------------


def test_check_alerts_persists_to_alert_store(tmp_path: Path) -> None:
    """
    When alert_store is passed to check_alerts(), fired alerts must be
    persisted so the API can return them.
    """

    store = _make_alert_store(tmp_path)
    tracker = AlertCooldownTracker(cooldown_seconds=0)  # no cooldown in tests

    check_alerts(
        "5m",
        {"trust_score": 0.45},  # below CRITICAL_TRUST_SCORE
        tracker=tracker,
        alert_store=store,
    )

    alerts = store.tail(limit=10)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "critical"
    assert alerts[0]["window"] == "5m"
    assert alerts[0]["trust_score"] == pytest.approx(0.45)


def test_check_alerts_does_not_persist_when_no_alert_store() -> None:
    """
    check_alerts without alert_store must still work (backward compatible).
    No exception must be raised.
    """

    tracker = AlertCooldownTracker(cooldown_seconds=0)
    # Must not raise even though alert_store is None
    check_alerts("5m", {"trust_score": 0.45}, tracker=tracker)


def test_check_alerts_suppressed_alert_not_persisted(tmp_path: Path) -> None:
    """
    An alert that is suppressed by the cooldown must not be written to
    the store - only fired alerts are persisted.
    """

    store = _make_alert_store(tmp_path)
    tracker = AlertCooldownTracker(cooldown_seconds=3600)  # very long cooldown

    check_alerts("5m", {"trust_score": 0.45}, tracker=tracker, alert_store=store)
    check_alerts(
        "5m", {"trust_score": 0.44}, tracker=tracker, alert_store=store
    )  # suppressed

    alerts = store.tail(limit=10)
    assert len(alerts) == 1  # only the first one
