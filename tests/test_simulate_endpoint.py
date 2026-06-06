"""Tests for POST /dashboard/decisions/simulate on the main branch.

Verifies that the endpoint:
- Returns HTTP 200 with mode='simulation'.
- Includes ``trust_components`` in the response (accuracy, f1, confidence,
  drift, latency) so the Streamlit dashboard can show the full breakdown.
- Returns a stable ``no_metrics_available`` payload when the store is empty.
- Uses ``DecisionStore.count()`` as batch_index (not a hardcoded 0).

The ``trust_components`` contract was added when the endpoint previously
discarded the second return value of ``compute_trust_score``.  Bug 33.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import model_monitor.api.dashboard as d_module
from model_monitor.api.main import app
from model_monitor.monitoring.types import MetricRecord
from model_monitor.storage.metrics_store import MetricsStore

client = TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path, **overrides: object) -> MetricsStore:
    """Write one MetricRecord to an isolated store and return the store."""
    store = MetricsStore(db_path=tmp_path / "metrics.db")
    record: MetricRecord = cast(
        MetricRecord,
        {
            "timestamp": time.time(),
            "batch_id": str(uuid.uuid4()),
            "n_samples": 128,
            "accuracy": 0.90,
            "f1": overrides.get("f1", 0.88),
            "avg_confidence": 0.85,
            "drift_score": overrides.get("drift_score", 0.02),
            "decision_latency_ms": 120.0,
            "calibration_error": None,
            "feature_drift_scores": None,
            "behavioral_violation_rate": None,
            "action": "none",
            "reason": "within thresholds",
            "previous_model": None,
            "new_model": None,
        },
    )
    store.write(record)
    return store


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


def test_simulate_returns_200(tmp_path: Path) -> None:
    """Endpoint returns HTTP 200 with mode='simulation'."""
    store = _make_store(tmp_path)
    with patch.object(d_module, "_metrics_store", store):
        resp = client.post("/dashboard/decisions/simulate")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "simulation"


def test_simulate_returns_trust_components(tmp_path: Path) -> None:
    """Response must include trust_components with all five performance signals.

    Previously the endpoint called ``compute_trust_score(..., _)`` - discarding
    the TrustScoreComponents second return value.  The Streamlit simulation
    panel cannot show per-signal breakdowns without it.  Bug 33.
    """
    store = _make_store(tmp_path)
    with patch.object(d_module, "_metrics_store", store):
        resp = client.post("/dashboard/decisions/simulate")

    data = resp.json()
    assert "trust_components" in data, (
        "trust_components missing from simulate response - "
        "Streamlit cannot show per-signal breakdown"
    )
    comps = data["trust_components"]
    for key in ("accuracy", "f1", "calibration", "drift", "latency", "data_quality"):
        assert key in comps, f"Missing trust component: {key!r}"
        assert 0.0 <= comps[key] <= 1.0, f"Component {key!r} out of [0,1]: {comps[key]}"


def test_simulate_action_and_reason_present(tmp_path: Path) -> None:
    """Response must include action and reason fields."""
    store = _make_store(tmp_path)
    with patch.object(d_module, "_metrics_store", store):
        resp = client.post("/dashboard/decisions/simulate")

    data = resp.json()
    assert "action" in data
    assert "reason" in data
    assert isinstance(data["action"], str)
    assert isinstance(data["reason"], str)


def test_simulate_inputs_contains_required_fields(tmp_path: Path) -> None:
    """The inputs dict must expose f1, f1_baseline, drift_score, batch_index."""
    store = _make_store(tmp_path, f1=0.91, drift_score=0.05)
    with patch.object(d_module, "_metrics_store", store):
        resp = client.post("/dashboard/decisions/simulate")

    inputs = resp.json()["inputs"]
    assert inputs["f1"] == pytest.approx(0.91)
    assert inputs["drift_score"] == pytest.approx(0.05)
    assert "f1_baseline" in inputs
    assert "batch_index" in inputs


def test_simulate_no_metrics_returns_no_metrics_available(tmp_path: Path) -> None:
    """Empty store → no_metrics_available reason without raising."""
    empty = MetricsStore(db_path=tmp_path / "empty.db")
    with patch.object(d_module, "_metrics_store", empty):
        resp = client.post("/dashboard/decisions/simulate")

    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "none"
    assert data["reason"] == "no_metrics_available"


def test_simulate_high_drift_raises_trust_score(tmp_path: Path) -> None:
    """High PSI drift lowers trust score relative to a stable record.

    Verifies that drift_score flows into compute_trust_score correctly inside
    simulate_decision - not just that the field appears in the response.
    """
    stable_store = _make_store(tmp_path / "stable", drift_score=0.01)
    drifted_store = _make_store(tmp_path / "drifted", drift_score=0.80)

    with patch.object(d_module, "_metrics_store", stable_store):
        stable_score = client.post("/dashboard/decisions/simulate").json()[
            "trust_score"
        ]

    with patch.object(d_module, "_metrics_store", drifted_store):
        drifted_score = client.post("/dashboard/decisions/simulate").json()[
            "trust_score"
        ]

    assert drifted_score < stable_score, (
        f"High drift should lower trust score: drifted={drifted_score:.4f} "
        f"stable={stable_score:.4f}"
    )
