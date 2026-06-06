"""Tests for health, readiness, and ingest endpoints.

These routes are simple but their failure modes are production-critical:
- ``GET /health`` must return 200 regardless of model or database state.
- ``GET /ready`` must return 200 when a model is loaded, 503 when absent.
- ``POST /metrics/ingest`` must enforce the API key when configured.

Coverage previously excluded ``api/*`` entirely.  These tests close the
gap on the endpoints that were excluded because they were "tested via HTTP
client on behavior-monitoring" - a justification that applied when the two
branches shared no API tests; it no longer holds.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from model_monitor.api.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_returns_200() -> None:
    """/health must always return 200 - even when no model is trained."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_is_idempotent() -> None:
    """Multiple /health calls must all return 200 (no side effects)."""
    for _ in range(3):
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def test_ready_returns_not_ready_when_no_model(tmp_path: Path) -> None:
    """/ready returns ready=False when no model file exists."""
    import model_monitor.api.health as h_module

    original = h_module._model_store
    from model_monitor.storage.model_store import ModelStore

    h_module._model_store = ModelStore(base_path=str(tmp_path))
    try:
        resp = client.get("/ready")
    finally:
        h_module._model_store = original

    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is False
    assert "reason" in data


def test_ready_returns_200_when_model_loaded(tmp_path: Path) -> None:
    """/ready returns ready=True when a valid model can be loaded."""
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    import model_monitor.api.health as h_module
    from model_monitor.storage.model_store import ModelStore

    # Train a minimal model and save it
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = (X[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=2, random_state=0).fit(X, y)
    model_path = tmp_path / "models" / "current.pkl"
    model_path.parent.mkdir(parents=True)
    joblib.dump(model, model_path)

    original = h_module._model_store
    store = ModelStore(base_path=str(tmp_path))
    h_module._model_store = store
    try:
        resp = client.get("/ready")
    finally:
        h_module._model_store = original

    assert resp.status_code == 200
    assert resp.json()["ready"] is True


# ---------------------------------------------------------------------------
# Ingest endpoint authentication
# ---------------------------------------------------------------------------


def test_ingest_returns_503_when_key_not_configured() -> None:
    """When MONITOR_API_KEY is not set the ingest endpoint returns 503.

    503 (not 401) because the service is administratively disabled -
    the endpoint has not been configured for production use.
    """
    env_key = "MONITOR_API_KEY"
    os.environ.pop(env_key, None)
    resp = client.post(
        "/metrics/ingest",
        json={
            "batch_id": "b0",
            "n_samples": 10,
            "accuracy": 0.9,
            "f1": 0.88,
            "avg_confidence": 0.85,
            "drift_score": 0.05,
            "decision_latency_ms": 100.0,
            "predictions": [1, 0] * 5,
        },
    )
    assert resp.status_code == 503


def test_ingest_returns_401_with_wrong_key() -> None:
    """Wrong API key returns 401."""
    os.environ["MONITOR_API_KEY"] = "correct-key"
    try:
        resp = client.post(
            "/metrics/ingest",
            headers={"X-Api-Key": "wrong-key"},
            json={
                "batch_id": "b0",
                "n_samples": 10,
                "accuracy": 0.9,
                "f1": 0.88,
                "avg_confidence": 0.85,
                "drift_score": 0.05,
                "decision_latency_ms": 100.0,
                "predictions": [1, 0] * 5,
            },
        )
    finally:
        os.environ.pop("MONITOR_API_KEY", None)
    assert resp.status_code == 401


def test_ingest_returns_401_with_no_key_when_configured() -> None:
    """Missing X-Api-Key header returns 401 when auth is enabled."""
    os.environ["MONITOR_API_KEY"] = "some-key"
    try:
        resp = client.post(
            "/metrics/ingest",
            json={
                "batch_id": "b0",
                "n_samples": 10,
                "accuracy": 0.9,
                "f1": 0.88,
                "avg_confidence": 0.85,
                "drift_score": 0.05,
                "decision_latency_ms": 100.0,
                "predictions": [1, 0] * 5,
            },
        )
    finally:
        os.environ.pop("MONITOR_API_KEY", None)
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Prometheus /metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_200() -> None:
    """/metrics must return 200 even when all stores are empty."""
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_endpoint_content_type_is_prometheus() -> None:
    """Content-Type must be the Prometheus text exposition type."""
    from prometheus_client import CONTENT_TYPE_LATEST

    resp = client.get("/metrics")
    assert resp.headers["content-type"] == CONTENT_TYPE_LATEST


def test_metrics_endpoint_contains_help_lines() -> None:
    """/metrics body must contain HELP lines for registered metrics."""
    resp = client.get("/metrics")
    body = resp.text
    assert "# HELP model_monitor_trust_score" in body
    assert "# HELP model_monitor_decisions_total" in body
