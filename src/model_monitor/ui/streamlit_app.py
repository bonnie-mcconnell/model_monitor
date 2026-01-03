from __future__ import annotations

import requests
import streamlit as st
import pandas as pd
from typing import Any

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 2.0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_get(path: str) -> dict[str, Any] | None:
    """
    Perform a GET request and return JSON if successful.
    Fails gracefully for UI safety.
    """
    try:
        resp = requests.get(
            f"{API_URL}{path}",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {path}")
        st.caption(str(exc))
        return None


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

col1, col2 = st.columns(2)

with col1:
    health = safe_get("/health")
    if health:
        st.success("Service healthy")
        st.json(health)
    else:
        st.warning("Health endpoint unavailable")

with col2:
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

if latest and "status" not in latest:
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{latest['accuracy']:.4f}")
    m2.metric("F1 Score", f"{latest['f1']:.4f}")
    m3.metric("Avg Confidence", f"{latest['avg_confidence']:.4f}")
    m4.metric("Drift Score", f"{latest['drift_score']:.4f}")
else:
    st.info("No metrics recorded yet.")


st.subheader("Operational Decisions")

decision_summary = safe_get("/metrics/decisions/summary")

if decision_summary:
    c1, c2, c3 = st.columns(3)
    c1.metric("Retrains", decision_summary.get("retrain", 0))
    c2.metric("Rollbacks", decision_summary.get("rollback", 0))
    c3.metric("Rejects", decision_summary.get("reject", 0))


# ---------------------------------------------------------------------
# Metrics history
# ---------------------------------------------------------------------
st.subheader("Metrics History")

history = safe_get("/metrics/tail?limit=500")

if history:
    df = pd.DataFrame(history)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Model Performance**")
            st.line_chart(
                df.set_index("timestamp")[["accuracy", "f1"]]
            )

        with c2:
            st.markdown("**Drift Signal**")
            st.line_chart(
                df.set_index("timestamp")["drift_score"]
            )
    else:
        st.info("Metrics history is empty.")
else:
    st.warning("Unable to load metrics history.")


decisions = safe_get("/metrics/decisions/tail?limit=200")

if decisions:
    ddf = pd.DataFrame(decisions)
    if not ddf.empty:
        ddf["timestamp"] = pd.to_datetime(ddf["timestamp"], unit="s")

        rollback_ts = ddf[ddf["action"] == "rollback"]["timestamp"]

        for ts in rollback_ts:
            st.caption(f"🔴 Rollback at {ts}")


st.subheader("Decision History")

if decisions:
    st.dataframe(
        pd.DataFrame(decisions).sort_values("timestamp", ascending=False),
        use_container_width=True,
    )
else:
    st.info("No decisions recorded yet.")


# VERSION DISPLAY

st.subheader("Active Model")

model_info = safe_get("/model/active")
if model_info:
    st.metric(
        "Active Version",
        model_info["version"],
        delta=model_info.get("previous_version"),
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.divider()
st.caption(
    "Model Monitor • Streaming inference, drift detection, retraining decisions"
)
