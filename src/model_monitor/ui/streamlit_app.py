"""Streamlit monitoring dashboard - connects to the FastAPI backend."""
from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st

from model_monitor.ui.decision_explanation import (
    decision_from_api,
    format_decision_explanation,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"
DASHBOARD = f"{API_BASE}/dashboard"
REQUEST_TIMEOUT = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_get(url: str) -> Any | None:
    """
    GET a JSON endpoint and return the parsed body.
    Returns None and renders an error on any failure.
    """
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {url}")
        st.caption(str(exc))
        return None


def coerce_timestamp(series: pd.Series) -> pd.Series:
    """
    Normalise timestamps that may be ISO strings or UNIX epoch floats.
    """
    as_unix = pd.to_datetime(series, errors="coerce", unit="s")
    as_iso = pd.to_datetime(series, errors="coerce")
    return as_unix.fillna(as_iso)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Model Monitor", layout="wide")
st.title("Model Monitoring Dashboard")
st.caption("Drift-aware, hysteresis-protected model governance")

# ---------------------------------------------------------------------------
# Service status
# ---------------------------------------------------------------------------

st.subheader("Service Status")

col_health, col_ready = st.columns(2)

with col_health:
    health = safe_get(f"{API_BASE}/health")
    if health:
        st.success("Service healthy")
        st.json(health)
    else:
        st.warning("Health endpoint unavailable")

with col_ready:
    ready = safe_get(f"{API_BASE}/ready")
    if ready:
        if ready.get("ready"):
            st.success("Model loaded and ready")
        else:
            st.warning(f"Not ready: {ready.get('reason', 'unknown')}")
        st.json(ready)
    else:
        st.warning("Readiness endpoint unavailable")

# ---------------------------------------------------------------------------
# Latest metrics
# ---------------------------------------------------------------------------

st.subheader("Latest Metrics")

latest = safe_get(f"{DASHBOARD}/metrics/latest")

if isinstance(latest, dict) and "accuracy" in latest:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",       f"{latest['accuracy']:.4f}")
    m2.metric("F1 Score",       f"{latest['f1']:.4f}")
    m3.metric("Avg Confidence", f"{latest['avg_confidence']:.4f}")
    m4.metric("Drift Score",    f"{latest['drift_score']:.4f}")
    m5.metric("Latency (ms)",   f"{latest['decision_latency_ms']:.1f}")
else:
    st.info("No metrics recorded yet. Run `make sim` to populate.")

# ---------------------------------------------------------------------------
# Metrics history charts
# ---------------------------------------------------------------------------

st.subheader("Metrics History")

history = safe_get(f"{DASHBOARD}/metrics/tail?limit=500")

if isinstance(history, list) and history:
    df = pd.DataFrame(history)
    df["timestamp"] = coerce_timestamp(df["timestamp"])
    df = df.sort_values("timestamp")

    col_perf, col_drift = st.columns(2)

    with col_perf:
        st.markdown("**Model Performance**")
        st.line_chart(df.set_index("timestamp")[["accuracy", "f1"]])

    with col_drift:
        st.markdown("**Drift Signal**")
        st.line_chart(df.set_index("timestamp")["drift_score"])
else:
    st.info("Metrics history unavailable.")

# ---------------------------------------------------------------------------
# Aggregated summaries
# ---------------------------------------------------------------------------

st.subheader("Rolling Window Summaries")

cols = st.columns(3)
for col, window in zip(cols, ["5m", "1h", "24h"]):
    summary = safe_get(f"{DASHBOARD}/metrics/summary/{window}")
    with col:
        st.markdown(f"**{window} window**")
        if isinstance(summary, dict) and summary:
            st.metric("Trust Score", f"{summary.get('trust_score', 'n/a')}")
            st.metric("Avg F1",      f"{summary.get('avg_f1', 'n/a'):.4f}" if summary.get("avg_f1") is not None else "n/a")
            st.metric("Avg Drift",   f"{summary.get('avg_drift_score', 'n/a'):.4f}" if summary.get("avg_drift_score") is not None else "n/a")
        else:
            st.caption("No data yet")

# ---------------------------------------------------------------------------
# Decision history
# ---------------------------------------------------------------------------

st.subheader("Decision History")

decisions = safe_get(f"{DASHBOARD}/decisions/history?limit=200")

if isinstance(decisions, list) and decisions:
    ddf = pd.DataFrame(decisions)
    if "timestamp" in ddf.columns:
        ddf["timestamp"] = coerce_timestamp(ddf["timestamp"])
        ddf = ddf.sort_values("timestamp", ascending=False)

    st.dataframe(ddf, use_container_width=True)

    st.markdown("#### Recent Decisions Explained")
    for payload in ddf.head(5).to_dict(orient="records"):
        try:
            decision = decision_from_api(payload)
            explanation = format_decision_explanation(decision)
            st.info(
                f"**{explanation['title']}** - {explanation['reason']}\n\n"
                f"{explanation['details']}"
            )
        except (KeyError, TypeError):
            continue
else:
    st.info("No decisions recorded yet.")

# ---------------------------------------------------------------------------
# Decision simulation
# ---------------------------------------------------------------------------

st.subheader("Decision Simulation")
st.caption(
    "Runs the decision engine against current metric summaries "
    "with no side effects."
)

if st.button("Simulate next decision"):
    try:
        resp = requests.post(
            f"{DASHBOARD}/decisions/simulate",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        st.success(
            f"**{result.get('action', 'unknown').upper()}** - "
            f"{result.get('reason', '')}"
        )
        st.json(result)
    except requests.RequestException as exc:
        st.error("Simulation request failed")
        st.caption(str(exc))

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption("model_monitor · github.com/bonnie-mcconnell/model_monitor")
