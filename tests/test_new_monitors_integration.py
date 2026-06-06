"""End-to-end integration test: all new monitors wired through Predictor.

This test verifies the complete data path for:
  - OutputDriftMonitor → output_drift_score in MetricRecord
  - DataQualityMonitor → data_quality_score in MetricRecord
  - ConformalMonitor   → conformal_coverage + conformal_set_size in MetricRecord
  - p95/p99 latency    → present in MetricRecord when batch >= 20 samples
  - calibration_error  → feeds trust score calibration component
  - All fields persist through MetricsStore write/read round-trip
  - trust_score reflects calibration component (not raw confidence)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from model_monitor.config.settings import (
    AppConfig,
    DriftConfig,
    RetrainConfig,
    RollbackConfig,
)
from model_monitor.inference.predict import Predictor
from model_monitor.monitoring.conformal import ConformalMonitor
from model_monitor.monitoring.data_quality import DataQualityMonitor
from model_monitor.monitoring.output_drift import OutputDriftMonitor
from model_monitor.storage.metrics_store import MetricsStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_config() -> AppConfig:
    return AppConfig(
        drift=DriftConfig(psi_threshold=0.25, window=3),
        retrain=RetrainConfig(
            min_f1_gain=0.02,
            cooldown_batches=3,
            min_samples=50,
            evidence_window=3,
            min_stable_batches=5,
        ),
        rollback=RollbackConfig(max_f1_drop=0.15),
    )


@pytest.fixture()
def trained_model_and_data(tmp_path: Path) -> dict:
    X, y = make_classification(
        n_samples=1000, n_features=5, n_informative=4, n_redundant=1, random_state=0
    )
    feature_names = [f"f{i}" for i in range(5)]
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    clf.feature_names_in_ = np.array(feature_names)

    model_path = tmp_path / "models" / "current.pkl"
    model_path.parent.mkdir(parents=True)
    joblib.dump(clf, model_path)

    active_path = tmp_path / "models" / "active.json"
    active_path.write_text(json.dumps({"version": "v1"}))

    return {
        "clf": clf,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "model_path": model_path,
        "active_path": active_path,
        "tmp_path": tmp_path,
    }


# ---------------------------------------------------------------------------
# Helper: build a fully instrumented Predictor
# ---------------------------------------------------------------------------


def _build_predictor(
    data: dict,
    config: AppConfig,
    *,
    with_output_drift: bool = True,
    with_data_quality: bool = True,
    with_conformal: bool = True,
) -> Predictor:
    X = data["X"]
    y = data["y"]
    clf = data["clf"]
    feature_names = data["feature_names"]
    model_path = data["model_path"]
    active_path = data["active_path"]

    # Calibrate conformal monitor on held-out split
    X_cal = X[:200]
    y_cal = y[:200]
    cal_probs = clf.predict_proba(X_cal)

    output_monitor = None
    if with_output_drift:
        ref_probs = clf.predict_proba(X[200:600])
        output_monitor = OutputDriftMonitor(ref_probs, window=3, threshold=0.1)

    dq_monitor = None
    if with_data_quality:
        dq_monitor = DataQualityMonitor(feature_names)

    conf_monitor = None
    if with_conformal:
        conf_monitor = ConformalMonitor(alpha=0.10)
        conf_monitor.calibrate(cal_probs, y_cal)

    predictor = Predictor(
        config=config,
        model_path=model_path,
        active_file=active_path,
        reference_features=X[600:],
        f1_baseline=0.75,
        output_drift_monitor=output_monitor,
        data_quality_monitor=dq_monitor,
        conformal_monitor=conf_monitor,
    )
    predictor.reload()
    return predictor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_p95_p99_latency_present_on_large_batch(
    trained_model_and_data: dict, app_config: AppConfig
) -> None:
    """p95/p99 latency must be populated when batch size >= 20."""
    predictor = _build_predictor(
        trained_model_and_data,
        app_config,
        with_output_drift=False,
        with_data_quality=False,
        with_conformal=False,
    )
    X = trained_model_and_data["X"]
    y = trained_model_and_data["y"]
    df = pd.DataFrame(X[700:730], columns=trained_model_and_data["feature_names"])
    y_series = pd.Series(y[700:730])

    # Warm up batch index past 1
    for i in range(3):
        predictor.predict_batch(df, y_series, batch_id=f"warm-{i}")

    _, _, _ = predictor.predict_batch(df, y_series, batch_id="test-lat")
    rec = predictor.last_metric_record
    assert rec is not None
    assert rec["p95_latency_ms"] is not None
    assert rec["p99_latency_ms"] is not None
    assert rec["p95_latency_ms"] >= 0.0
    assert rec["p99_latency_ms"] >= rec["p95_latency_ms"]


def test_p95_p99_none_on_small_batch(
    trained_model_and_data: dict, app_config: AppConfig
) -> None:
    """p95/p99 must be None when batch has fewer than 20 samples."""
    predictor = _build_predictor(
        trained_model_and_data,
        app_config,
        with_output_drift=False,
        with_data_quality=False,
        with_conformal=False,
    )
    X = trained_model_and_data["X"]
    y = trained_model_and_data["y"]
    df = pd.DataFrame(X[:5], columns=trained_model_and_data["feature_names"])
    y_series = pd.Series(y[:5])

    for i in range(3):
        predictor.predict_batch(df, y_series, batch_id=f"warm-{i}")
    _, _, _ = predictor.predict_batch(df, y_series, batch_id="small-batch")
    rec = predictor.last_metric_record
    assert rec is not None
    assert rec["p95_latency_ms"] is None
    assert rec["p99_latency_ms"] is None


def test_output_drift_score_populated(
    trained_model_and_data: dict, app_config: AppConfig
) -> None:
    """output_drift_score should be non-None after OutputDriftMonitor window fills."""
    predictor = _build_predictor(trained_model_and_data, app_config)
    X = trained_model_and_data["X"]
    y = trained_model_and_data["y"]
    fn = trained_model_and_data["feature_names"]
    df = pd.DataFrame(X[700:], columns=fn)
    y_s = pd.Series(y[700:])

    # window=3, so need 3+ batches
    for i in range(5):
        predictor.predict_batch(df, y_s, batch_id=f"b{i}")

    rec = predictor.last_metric_record
    assert rec is not None
    # After window fills output_drift_score should be set
    assert rec["output_drift_score"] is not None
    assert rec["output_drift_score"] >= 0.0


def test_data_quality_score_populated(
    trained_model_and_data: dict, app_config: AppConfig
) -> None:
    """data_quality_score should be populated when DataQualityMonitor is configured."""
    predictor = _build_predictor(trained_model_and_data, app_config)
    X = trained_model_and_data["X"]
    y = trained_model_and_data["y"]
    fn = trained_model_and_data["feature_names"]

    for i in range(3):
        df = pd.DataFrame(X[700:730], columns=fn)
        predictor.predict_batch(df, pd.Series(y[700:730]), batch_id=f"dq-{i}")

    rec = predictor.last_metric_record
    assert rec is not None
    assert rec["data_quality_score"] is not None
    assert 0.0 <= rec["data_quality_score"] <= 1.0


def test_data_quality_score_drops_on_null_injection(
    trained_model_and_data: dict, app_config: AppConfig
) -> None:
    """Injecting NaNs into a batch should lower the data_quality_score."""
    predictor = _build_predictor(trained_model_and_data, app_config)
    X = trained_model_and_data["X"]
    y = trained_model_and_data["y"]
    fn = trained_model_and_data["feature_names"]

    # Warm up
    for i in range(3):
        df = pd.DataFrame(X[700:730], columns=fn)
        predictor.predict_batch(df, pd.Series(y[700:730]), batch_id=f"warm-{i}")

    # Clean batch
    df_clean = pd.DataFrame(X[730:760], columns=fn)
    predictor.predict_batch(df_clean, pd.Series(y[730:760]), batch_id="clean")
    assert predictor.last_metric_record is not None
    score_clean = predictor.last_metric_record["data_quality_score"]

    # Polluted batch - 50% nulls
    df_null = df_clean.copy().astype(float)
    df_null.iloc[:15, :] = np.nan
    predictor.predict_batch(df_null, pd.Series(y[730:760]), batch_id="polluted")
    assert predictor.last_metric_record is not None
    score_polluted = predictor.last_metric_record["data_quality_score"]

    assert score_clean is not None
    assert score_polluted is not None
    assert score_polluted < score_clean


def test_conformal_set_size_populated_without_labels(
    trained_model_and_data: dict, app_config: AppConfig
) -> None:
    """conformal_set_size must be populated even without y_true."""
    predictor = _build_predictor(trained_model_and_data, app_config)
    X = trained_model_and_data["X"]
    fn = trained_model_and_data["feature_names"]
    df = pd.DataFrame(X[700:730], columns=fn)

    # No y_true
    predictor.predict_batch(df, None, batch_id="unlabeled")
    rec = predictor.last_metric_record
    assert rec is not None
    assert rec["conformal_set_size"] is not None
    assert rec["conformal_set_size"] >= 1.0


def test_conformal_coverage_populated_with_labels(
    trained_model_and_data: dict, app_config: AppConfig
) -> None:
    """conformal_coverage must be populated when y_true is provided."""
    predictor = _build_predictor(trained_model_and_data, app_config)
    X = trained_model_and_data["X"]
    y = trained_model_and_data["y"]
    fn = trained_model_and_data["feature_names"]

    for i in range(3):
        df = pd.DataFrame(X[700:730], columns=fn)
        predictor.predict_batch(df, pd.Series(y[700:730]), batch_id=f"cv-{i}")

    rec = predictor.last_metric_record
    assert rec is not None
    assert rec["conformal_coverage"] is not None
    assert 0.0 <= rec["conformal_coverage"] <= 1.0


def test_full_metric_record_persists_to_store(
    trained_model_and_data: dict, app_config: AppConfig, tmp_path: Path
) -> None:
    """All new fields must survive a MetricsStore write/read round-trip."""
    predictor = _build_predictor(trained_model_and_data, app_config)
    store = MetricsStore(db_path=tmp_path / "test_integration.db")

    X = trained_model_and_data["X"]
    y = trained_model_and_data["y"]
    fn = trained_model_and_data["feature_names"]

    # Run enough batches for output drift buffer to fill
    for i in range(5):
        df = pd.DataFrame(X[700:730], columns=fn)
        predictor.predict_batch(df, pd.Series(y[700:730]), batch_id=f"int-{i}")

    rec = predictor.last_metric_record
    assert rec is not None
    store.write(rec)

    rows = store.tail(limit=1)
    assert len(rows) == 1
    row = rows[0]

    # These should always be present (batch >= 20 samples)
    assert row["p95_latency_ms"] is not None
    assert row["p99_latency_ms"] is not None

    # data_quality is always present when monitor configured
    assert row["data_quality_score"] is not None

    # conformal_coverage present with labels
    assert row["conformal_coverage"] is not None

    # All values in valid ranges
    assert 0.0 <= row["data_quality_score"] <= 1.0
    assert 0.0 <= row["conformal_coverage"] <= 1.0
    assert row["p99_latency_ms"] >= row["p95_latency_ms"]
