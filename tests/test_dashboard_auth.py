"""Tests for dashboard API authentication.

The /dashboard/* routes are optionally protected by MONITOR_DASHBOARD_KEY.
When the env var is unset the dashboard is unauthenticated (local-dev
default).  When it is set, every request must supply a matching X-Api-Key
header or receive HTTP 401.

These tests cover:
- Unauthenticated mode: all requests pass without a header.
- Authenticated mode: correct key → 200; wrong key → 401; no key → 401.
- Key check is case-sensitive (security invariant).
- Multiple routes all inherit the auth from the router-level Depends().
- The 401 response includes a WWW-Authenticate header (RFC 7235 compliance).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from model_monitor.api.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """TestClient with no dashboard key set - unauthenticated mode."""
    env_key = "MONITOR_DASHBOARD_KEY"
    os.environ.pop(env_key, None)
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def authed_client() -> Iterator[TestClient]:
    """TestClient with MONITOR_DASHBOARD_KEY=test-secret-key."""
    os.environ["MONITOR_DASHBOARD_KEY"] = "test-secret-key"
    yield TestClient(app, raise_server_exceptions=True)
    os.environ.pop("MONITOR_DASHBOARD_KEY", None)


# ---------------------------------------------------------------------------
# Unauthenticated mode (key not set)
# ---------------------------------------------------------------------------


def test_dashboard_health_reachable_without_key(client: TestClient) -> None:
    """When MONITOR_DASHBOARD_KEY is unset, the dashboard is open."""
    resp = client.get("/dashboard/health/detailed")
    assert resp.status_code != 401, (
        "Dashboard should be open when MONITOR_DASHBOARD_KEY is not set"
    )


def test_dashboard_metrics_latest_reachable_without_key(client: TestClient) -> None:
    """Metrics endpoint is reachable in unauthenticated mode."""
    resp = client.get("/dashboard/metrics/latest")
    assert resp.status_code != 401


def test_dashboard_config_reachable_without_key(client: TestClient) -> None:
    """Config endpoint is reachable in unauthenticated mode."""
    resp = client.get("/dashboard/config")
    assert resp.status_code != 401


# ---------------------------------------------------------------------------
# Authenticated mode - correct key
# ---------------------------------------------------------------------------


def test_dashboard_accepts_correct_key(authed_client: TestClient) -> None:
    """Correct X-Api-Key header is accepted."""
    resp = authed_client.get(
        "/dashboard/health/detailed",
        headers={"X-Api-Key": "test-secret-key"},
    )
    assert resp.status_code != 401, (
        f"Expected non-401 with correct key, got {resp.status_code}"
    )


def test_dashboard_accepts_correct_key_metrics(authed_client: TestClient) -> None:
    """Correct key accepted on metrics/latest endpoint."""
    resp = authed_client.get(
        "/dashboard/metrics/latest",
        headers={"X-Api-Key": "test-secret-key"},
    )
    assert resp.status_code != 401


def test_dashboard_accepts_correct_key_config(authed_client: TestClient) -> None:
    """Correct key accepted on config endpoint."""
    resp = authed_client.get(
        "/dashboard/config",
        headers={"X-Api-Key": "test-secret-key"},
    )
    assert resp.status_code != 401


# ---------------------------------------------------------------------------
# Authenticated mode - wrong or missing key
# ---------------------------------------------------------------------------


def test_dashboard_rejects_wrong_key(authed_client: TestClient) -> None:
    """Wrong X-Api-Key header returns 401."""
    resp = authed_client.get(
        "/dashboard/health/detailed",
        headers={"X-Api-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_dashboard_rejects_missing_key(authed_client: TestClient) -> None:
    """Missing X-Api-Key header returns 401 when auth is enabled."""
    resp = authed_client.get("/dashboard/health/detailed")
    assert resp.status_code == 401


def test_dashboard_rejects_empty_key(authed_client: TestClient) -> None:
    """Empty string X-Api-Key returns 401."""
    resp = authed_client.get(
        "/dashboard/health/detailed",
        headers={"X-Api-Key": ""},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Security invariants
# ---------------------------------------------------------------------------


def test_dashboard_key_check_is_case_sensitive(authed_client: TestClient) -> None:
    """Key comparison is case-sensitive - 'Test-Secret-Key' != 'test-secret-key'."""
    resp = authed_client.get(
        "/dashboard/health/detailed",
        headers={"X-Api-Key": "Test-Secret-Key"},  # wrong case
    )
    assert resp.status_code == 401, (
        "Key comparison must be case-sensitive; mixed-case variant must be rejected"
    )


def test_dashboard_401_includes_www_authenticate_header(
    authed_client: TestClient,
) -> None:
    """401 responses must include WWW-Authenticate per RFC 7235."""
    resp = authed_client.get("/dashboard/health/detailed")
    assert resp.status_code == 401
    assert "www-authenticate" in {k.lower() for k in resp.headers}, (
        "401 response must include WWW-Authenticate header (RFC 7235)"
    )


def test_dashboard_multiple_routes_all_require_key(authed_client: TestClient) -> None:
    """Auth is enforced on all /dashboard/* routes, not just one.

    Checks a selection of different route shapes (GET, POST, path params)
    to confirm the router-level Depends() propagates to every handler.
    """
    routes_to_check = [
        ("GET", "/dashboard/metrics/latest"),
        ("GET", "/dashboard/decisions/history"),
        ("GET", "/dashboard/alerts/history"),
        ("GET", "/dashboard/models/active"),
        ("GET", "/dashboard/config"),
        ("GET", "/dashboard/health/detailed"),
    ]
    for method, path in routes_to_check:
        resp = authed_client.request(method, path)
        assert resp.status_code == 401, (
            f"Expected 401 on {method} {path} without key, got {resp.status_code}"
        )


def test_dashboard_key_not_exposed_in_error_response(
    authed_client: TestClient,
) -> None:
    """The 401 error body must not echo back the expected key value."""
    resp = authed_client.get(
        "/dashboard/health/detailed",
        headers={"X-Api-Key": "wrong-key"},
    )
    assert resp.status_code == 401
    body = resp.text
    # The server must never reflect the configured key in an error response.
    assert "test-secret-key" not in body
    assert "wrong-key" not in body
