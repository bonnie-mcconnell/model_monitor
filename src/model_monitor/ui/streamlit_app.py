"""Streamlit monitoring dashboard - connects to the FastAPI backend."""
from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st
from ui.decision_explanation import decision_from_api, format_decision_explanation

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 2.0

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_get(path: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    try:
        resp = requests.get(
            f"{API_URL}{path}",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result: dict[str, Any] | list[dict[str, Any]] = resp.json()
        return result
    except requests.RequestException as exc:
        st.error(f"API request failed: {path}")
        st.caption(str(exc))
        return None


def coerce_timestamp(series: pd.Series) -> pd.Series:
    """
    Normalize timestamps that may be ISO or UNIX seconds.
    """
    return pd.to_datetime(series, errors="coerce", unit="s").fillna(
        pd.to_datetime(series, errors="coerce")
    )


# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Model Monitor",
    layout="wide",
)

st.title("📊 Model Monitoring Dashboard")
st.caption("Real-time model health, performance, and drift signals")

# ---------------------------------------------------------------------
# Service status
# ---------------------------------------------------------------------
st.subheader("Service Status")

c1, c2 = st.columns(2)

with c1:
    health = safe_get("/health")
    if health:
        st.success("Service healthy")
        st.json(health)
    else:
        st.warning("Health endpoint unavailable")

with c2:
    ready = safe_get("/ready")
    if ready:
        st.success("Service ready")
        st.json(ready)
    else:
        st.warning("Readiness endpoint unavailable")

# ---------------------------------------------------------------------
# Latest metrics
# ---------------------------------------------------------------------
st.subheader("Latest Metrics")

latest = safe_get("/metrics/latest")

if isinstance(latest, dict) and "accuracy" in latest:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{latest['accuracy']:.4f}")
    m2.metric("F1 Score", f"{latest['f1']:.4f}")
    m3.metric("Avg Confidence", f"{latest['avg_confidence']:.4f}")
    m4.metric("Drift Score", f"{latest['drift_score']:.4f}")
else:
    st.info("No metrics recorded yet.")

# ---------------------------------------------------------------------
# Metrics history
# ---------------------------------------------------------------------
st.subheader("Metrics History")

history = safe_get("/metrics/tail?limit=500")

if isinstance(history, list) and history:
    df = pd.DataFrame(history)
    df["timestamp"] = coerce_timestamp(df["timestamp"])
    df = df.sort_values("timestamp")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Model Performance**")
        st.line_chart(df.set_index("timestamp")[["accuracy", "f1"]])

    with c2:
        st.markdown("**Drift Signal**")
        st.line_chart(df.set_index("timestamp")["drift_score"])
else:
    st.info("Metrics history unavailable.")

# ---------------------------------------------------------------------
# Decision history
# ---------------------------------------------------------------------
st.subheader("Decision History")

decisions = safe_get("/metrics/decisions/tail?limit=200")

if isinstance(decisions, list) and decisions:
    ddf = pd.DataFrame(decisions)
    ddf["timestamp"] = coerce_timestamp(ddf["timestamp"])
    ddf = ddf.sort_values("timestamp", ascending=False)

    st.dataframe(ddf, use_container_width=True)

    st.markdown("### Recent Decisions Explained")


    for payload in ddf.head(5).to_dict(orient="records"):
        decision = decision_from_api(payload)
        explanation = format_decision_explanation(decision)

        st.info(
            f"**{explanation['title']}**\n\n"
            f"{explanation['reason']}\n\n"
            f"{explanation['details']}"
        )

else:
    st.info("No decisions recorded yet.")

# ---------------------------------------------------------------------
# Active model
# ---------------------------------------------------------------------
st.subheader("Active Model")

model_info = safe_get("/model/active")

if isinstance(model_info, dict):
    st.metric(
        "Active Version",
        model_info["version"],
        delta=model_info.get("previous_version"),
    )
else:
    st.info("No active model information available.")


# ---------------------------------------------------------------------
# Simulation Controls
# ---------------------------------------------------------------------
st.subheader("Decision Simulation")

simulate = st.toggle("Simulation mode (no side effects)", value=True)

if st.button("Evaluate next decision"):
    endpoint = "/decisions/simulate" if simulate else "/decisions/execute"

    try:
        resp = requests.post(
            f"{API_URL}{endpoint}",
            json={},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()

        explanation = format_decision_explanation(result)

        st.success(
            f"**{explanation['title']}**\n\n"
            f"{explanation['reason']}\n\n"
            f"{explanation['details']}"
        )

    except requests.RequestException as exc:
        st.error("Decision request failed")
        st.caption(str(exc))

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.divider()
st.caption(
    "Model Monitor • Drift-aware, hysteresis-protected model governance"
)
