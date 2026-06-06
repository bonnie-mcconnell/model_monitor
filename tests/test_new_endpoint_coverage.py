"""Tests for the two endpoints added in v7 that previously had zero coverage.

  GET /dashboard/causal-drift/latest
  GET /dashboard/threshold-advisor/status

Both endpoints must:
  - Return 200 unconditionally (never 404/500 even when no data exists).
  - Return a JSON object with an ``available`` key that is False when the
    feature is not configured or no data has been recorded.
  - Return the expected payload shape when data is present.

These tests use FastAPI's TestClient with a temporary SQLite store so they
run in-process without a live server, consistent with every other API test
in this suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from model_monitor.api.main import app
from model_monitor.core.decisions import DecisionType
from model_monitor.monitoring.threshold_advisor import ThresholdAdvisor
from model_monitor.monitoring.types import MetricRecord
from model_monitor.storage.metrics_store import MetricsStore

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_record(**extra: object) -> MetricRecord:
    """Return a complete valid MetricRecord, overridable via kwargs."""
    base: MetricRecord = {
        "batch_id": "b_test",
        "timestamp": 1_700_000_000.0,
        "n_samples": 100,
        "accuracy": 0.90,
        "f1": 0.88,
        "avg_confidence": 0.82,
        "drift_score": 0.05,
        "decision_latency_ms": 1.2,
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
        "causal_drift_report": None,
        "mmd_p_value": None,
        "mmd_is_drift": None,
        "shap_attribution": None,
        "action": DecisionType.NONE,
        "reason": "stable",
        "previous_model": None,
        "new_model": None,
    }
    base.update(extra)  # type: ignore[typeddict-item]
    return base


def _ready_advisor(n: int = 12) -> ThresholdAdvisor:
    """Return a ThresholdAdvisor with ``n`` stable-period observations recorded."""
    advisor = ThresholdAdvisor(
        feature_names=["f0", "f1"],
        alpha=0.05,
        min_batches=10,
    )
    rng = np.random.default_rng(42)
    for _ in range(n):
        advisor.observe(
            psi_scores=[float(rng.uniform(0, 0.04)), float(rng.uniform(0, 0.03))],
            trust_score=float(rng.uniform(0.82, 0.95)),
        )
    return advisor


# ---------------------------------------------------------------------------
# GET /dashboard/causal-drift/latest
# ---------------------------------------------------------------------------


class TestCausalDriftLatest:
    """Endpoint contract for GET /dashboard/causal-drift/latest."""

    def test_returns_200_with_empty_store(self, tmp_path: Path) -> None:
        """Returns 200 even when no batches have been recorded."""
        store = MetricsStore(db_path=str(tmp_path / "empty.db"))
        with patch("model_monitor.api.dashboard._get_metrics_store", return_value=store):
            resp = client.get("/dashboard/causal-drift/latest")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["available"] is False
        assert "reason" in payload

    def test_available_false_when_record_has_no_causal_report(
        self, tmp_path: Path
    ) -> None:
        """available=False when the latest record carries no causal_drift_report."""
        store = MetricsStore(db_path=str(tmp_path / "m.db"))
        store.write(_full_record())
        with patch("model_monitor.api.dashboard._get_metrics_store", return_value=store):
            resp = client.get("/dashboard/causal-drift/latest")
        assert resp.status_code == 200
        assert resp.json()["available"] is False

    def test_returns_report_when_causal_data_present(self, tmp_path: Path) -> None:
        """available=True and report payload returned when causal_drift_report is set."""
        store = MetricsStore(db_path=str(tmp_path / "m.db"))
        causal_payload = {
            "dominant_cause": "genuine_shift",
            "recommendation": "retrain on new data",
            "feature_results": [
                {"feature_name": "f0", "psi": 0.15, "drift_class": "genuine_shift"}
            ],
        }
        store.write(
            _full_record(
                drift_score=0.18,
                trust_score=0.65,
                causal_drift_report=json.dumps(causal_payload),
            )
        )
        with patch("model_monitor.api.dashboard._get_metrics_store", return_value=store):
            resp = client.get("/dashboard/causal-drift/latest")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["available"] is True
        assert "report" in payload


# ---------------------------------------------------------------------------
# GET /dashboard/threshold-advisor/status
# ---------------------------------------------------------------------------


class TestThresholdAdvisorStatus:
    """Endpoint contract for GET /dashboard/threshold-advisor/status."""

    def test_returns_200_when_no_advisor_configured(self) -> None:
        """Returns 200 with available=False when predictor has no threshold advisor."""
        # spec=[] means the MagicMock has no attributes - getattr returns AttributeError,
        # which is what the endpoint's getattr(..., None) guard is designed to handle.
        mock_pred = MagicMock(spec=[])
        mock_startup = MagicMock()
        mock_startup._predictor = mock_pred

        with patch.dict("sys.modules", {"model_monitor.api.startup": mock_startup}):
            resp = client.get("/dashboard/threshold-advisor/status")

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["available"] is False
        assert "reason" in payload

    def test_returns_200_when_predictor_absent(self) -> None:
        """Degrades gracefully when _predictor is None (cold start)."""
        mock_startup = MagicMock()
        mock_startup._predictor = None

        with patch.dict("sys.modules", {"model_monitor.api.startup": mock_startup}):
            resp = client.get("/dashboard/threshold-advisor/status")

        assert resp.status_code == 200
        assert resp.json()["available"] is False

    def test_status_below_min_batches(self) -> None:
        """n_observations and is_ready=False returned before threshold reached."""
        advisor = ThresholdAdvisor(
            feature_names=["f0", "f1"],
            alpha=0.05,
            min_batches=20,
        )
        rng = np.random.default_rng(7)
        for _ in range(5):
            advisor.observe(
                psi_scores=[float(rng.uniform(0, 0.05)), float(rng.uniform(0, 0.04))],
                trust_score=float(rng.uniform(0.80, 0.92)),
            )

        mock_pred = MagicMock()
        mock_pred._threshold_advisor = advisor
        mock_startup = MagicMock()
        mock_startup._predictor = mock_pred

        with patch.dict("sys.modules", {"model_monitor.api.startup": mock_startup}):
            resp = client.get("/dashboard/threshold-advisor/status")

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["available"] is True
        assert payload["n_observations"] == 5
        assert payload["is_ready"] is False
        # No recommendation yet - not enough stable batches.
        assert "recommendation" not in payload

    def test_recommendation_present_when_advisor_ready(self) -> None:
        """psi_warn_global and trust_warn present when advisor has enough data."""
        advisor = _ready_advisor(n=12)

        mock_pred = MagicMock()
        mock_pred._threshold_advisor = advisor
        mock_startup = MagicMock()
        mock_startup._predictor = mock_pred

        with patch.dict("sys.modules", {"model_monitor.api.startup": mock_startup}):
            resp = client.get("/dashboard/threshold-advisor/status")

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["available"] is True
        assert payload["is_ready"] is True
        assert "recommendation" in payload
        rec = payload["recommendation"]
        assert "psi_warn_global" in rec
        assert "trust_warn" in rec
        assert isinstance(rec["psi_warn_per_feature"], dict)
