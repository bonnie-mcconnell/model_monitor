"""Tests for schema migrations 6 and 7.

Verifies that the new columns are added by the migrations and that
the MetricsStore correctly round-trips the new fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model_monitor.storage.metrics_store import MetricsStore


def _full_record(batch_id: str = "b1") -> dict:
    """Return a MetricRecord dict with all optional new fields populated."""
    import time

    return {
        "timestamp": time.time(),
        "batch_id": batch_id,
        "n_samples": 50,
        "accuracy": 0.90,
        "f1": 0.88,
        "avg_confidence": 0.85,
        "drift_score": 0.05,
        "decision_latency_ms": 120.0,
        "p95_latency_ms": 180.0,
        "p99_latency_ms": 250.0,
        "calibration_error": 0.03,
        "feature_drift_scores": [0.01, 0.02, 0.03],
        "output_drift_score": 0.07,
        "output_drift_class_scores": [0.05, 0.09],
        "data_quality_score": 0.95,
        "conformal_coverage": 0.92,
        "conformal_set_size": 1.1,
        "behavioral_violation_rate": 0.02,
        "shap_attribution": {"f0": 0.01, "f1": -0.02},
        "action": "none",
        "reason": "test",
        "previous_model": None,
        "new_model": None,
    }


def test_new_columns_written_and_read_back(tmp_path: Path) -> None:
    """All new MetricRecord fields must survive a write-then-read round-trip."""
    store = MetricsStore(db_path=tmp_path / "metrics.db")
    record = _full_record()
    store.write(record)  # type: ignore[arg-type]

    rows = store.tail(limit=1)
    assert len(rows) == 1
    row = rows[0]

    assert row["p95_latency_ms"] == pytest.approx(180.0)
    assert row["p99_latency_ms"] == pytest.approx(250.0)
    assert row["output_drift_score"] == pytest.approx(0.07)
    assert row["output_drift_class_scores"] == pytest.approx([0.05, 0.09])
    assert row["data_quality_score"] == pytest.approx(0.95)
    assert row["conformal_coverage"] == pytest.approx(0.92)
    assert row["conformal_set_size"] == pytest.approx(1.1)


def test_null_new_fields_written_and_read_back(tmp_path: Path) -> None:
    """New optional fields must survive as None when not provided."""
    store = MetricsStore(db_path=tmp_path / "metrics.db")
    import time

    record = {
        "timestamp": time.time(),
        "batch_id": "null-test",
        "n_samples": 10,
        "accuracy": 0.80,
        "f1": 0.78,
        "avg_confidence": 0.75,
        "drift_score": 0.0,
        "decision_latency_ms": 50.0,
        "p95_latency_ms": None,
        "p99_latency_ms": None,
        "calibration_error": None,
        "feature_drift_scores": None,
        "output_drift_score": None,
        "output_drift_class_scores": None,
        "data_quality_score": None,
        "conformal_coverage": None,
        "conformal_set_size": None,
        "behavioral_violation_rate": None,
        "shap_attribution": None,
        "action": "none",
        "reason": "test",
        "previous_model": None,
        "new_model": None,
    }
    store.write(record)  # type: ignore[arg-type]

    rows = store.tail(limit=1)
    assert len(rows) == 1
    row = rows[0]
    for field in (
        "p95_latency_ms",
        "p99_latency_ms",
        "output_drift_score",
        "data_quality_score",
        "conformal_coverage",
        "conformal_set_size",
    ):
        assert row[field] is None, f"Expected None for {field}, got {row[field]}"


def test_migration_idempotent(tmp_path: Path) -> None:
    """Creating MetricsStore twice on the same DB must not raise."""
    db = tmp_path / "metrics.db"
    MetricsStore(db_path=db)
    MetricsStore(db_path=db)  # second creation: migrations are no-ops
