"""
Tests for POST /metrics/ingest.

Uses FastAPI's TestClient so these run without a live server.
The MONITOR_API_KEY environment variable is patched per test - no
global state is mutated between tests.
"""
from __future__ import annotations

import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from model_monitor.api.main import app

_ENV_KEY = "MONITOR_API_KEY"
_VALID_KEY = "test-secret-key-abc123"

client = TestClient(app, raise_server_exceptions=True)

_VALID_PAYLOAD = {
    "batch_id": "test-batch-001",
    "n_samples": 64,
    "accuracy": 0.91,
    "f1": 0.89,
    "avg_confidence": 0.84,
    "drift_score": 0.04,
    "decision_latency_ms": 122.5,
    "action": "none",
    "reason": "System operating within thresholds",
    "previous_model": None,
    "new_model": None,
}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def test_missing_api_key_returns_401() -> None:
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post("/metrics/ingest", json=_VALID_PAYLOAD)
    assert response.status_code == 401


def test_wrong_api_key_returns_401() -> None:
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post(
            "/metrics/ingest",
            json=_VALID_PAYLOAD,
            headers={"x-api-key": "wrong-key"},
        )
    assert response.status_code == 401


def test_endpoint_disabled_when_env_var_not_set() -> None:
    env_without_key = {k: v for k, v in os.environ.items() if k != _ENV_KEY}
    with patch.dict(os.environ, env_without_key, clear=True):
        response = client.post(
            "/metrics/ingest",
            json=_VALID_PAYLOAD,
            headers={"x-api-key": _VALID_KEY},
        )
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_request_returns_201() -> None:
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post(
            "/metrics/ingest",
            json=_VALID_PAYLOAD,
            headers={"x-api-key": _VALID_KEY},
        )
    assert response.status_code == 201


def test_valid_request_echoes_batch_id() -> None:
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post(
            "/metrics/ingest",
            json=_VALID_PAYLOAD,
            headers={"x-api-key": _VALID_KEY},
        )
    body = response.json()
    assert body["accepted"] is True
    assert body["batch_id"] == _VALID_PAYLOAD["batch_id"]
    assert "timestamp" in body


# ---------------------------------------------------------------------------
# Input validation - malformed payloads must return 422
# ---------------------------------------------------------------------------

def test_malformed_payload_missing_required_field_returns_422() -> None:
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "accuracy"}
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post(
            "/metrics/ingest",
            json=payload,
            headers={"x-api-key": _VALID_KEY},
        )
    assert response.status_code == 422


def test_malformed_payload_wrong_type_returns_422() -> None:
    payload = {**_VALID_PAYLOAD, "n_samples": "not-an-int"}
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post(
            "/metrics/ingest",
            json=payload,
            headers={"x-api-key": _VALID_KEY},
        )
    assert response.status_code == 422


def test_malformed_payload_invalid_action_returns_422() -> None:
    payload = {**_VALID_PAYLOAD, "action": "explode"}
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post(
            "/metrics/ingest",
            json=payload,
            headers={"x-api-key": _VALID_KEY},
        )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Dashboard decisions endpoint - verify clean serialisation
# ---------------------------------------------------------------------------

def test_decisions_history_does_not_leak_sa_instance_state() -> None:
    """
    The dashboard decisions/history endpoint previously used __dict__ on
    ORM objects, which leaks _sa_instance_state into the response.
    This test verifies the fix: only mapped column values are returned.
    """
    response = client.get("/dashboard/decisions/history?limit=5")
    assert response.status_code == 200
    records = response.json()
    for record in records:
        assert "_sa_instance_state" not in record, (
            "ORM internal state leaked into API response. "
            "Use _orm_to_dict() not __dict__."
        )


def test_dashboard_metrics_summary_does_not_leak_sa_instance_state() -> None:
    """
    Verify the metrics summary endpoint also uses _orm_to_dict.
    """
    response = client.get("/dashboard/metrics/summary/5m")
    # 200 if data exists, 404 if no data - either is fine for this test
    if response.status_code == 200:
        data = response.json()
        assert "_sa_instance_state" not in data


# ---------------------------------------------------------------------------
# Input validation - metric range bounds
# ---------------------------------------------------------------------------

def test_accuracy_above_one_returns_422() -> None:
    """
    accuracy > 1.0 must be rejected. The trust score formula assumes all
    performance metrics are in [0, 1] - accepting out-of-range values would
    silently corrupt the computed trust score.
    """
    payload = {**_VALID_PAYLOAD, "accuracy": 1.5}
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post(
            "/metrics/ingest",
            json=payload,
            headers={"x-api-key": _VALID_KEY},
        )
    assert response.status_code == 422


def test_f1_below_zero_returns_422() -> None:
    payload = {**_VALID_PAYLOAD, "f1": -0.1}
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post(
            "/metrics/ingest",
            json=payload,
            headers={"x-api-key": _VALID_KEY},
        )
    assert response.status_code == 422


def test_negative_latency_returns_422() -> None:
    payload = {**_VALID_PAYLOAD, "decision_latency_ms": -1.0}
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post(
            "/metrics/ingest",
            json=payload,
            headers={"x-api-key": _VALID_KEY},
        )
    assert response.status_code == 422


def test_zero_n_samples_returns_422() -> None:
    payload = {**_VALID_PAYLOAD, "n_samples": 0}
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post(
            "/metrics/ingest",
            json=payload,
            headers={"x-api-key": _VALID_KEY},
        )
    assert response.status_code == 422


def test_empty_batch_id_returns_422() -> None:
    payload = {**_VALID_PAYLOAD, "batch_id": ""}
    with patch.dict(os.environ, {_ENV_KEY: _VALID_KEY}):
        response = client.post(
            "/metrics/ingest",
            json=payload,
            headers={"x-api-key": _VALID_KEY},
        )
    assert response.status_code == 422
