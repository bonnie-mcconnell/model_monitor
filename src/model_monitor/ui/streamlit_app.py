"""Streamlit monitoring dashboard -  connects to the FastAPI backend."""
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

API_BASE        = "http://localhost:8000"
DASHBOARD       = f"{API_BASE}/dashboard"
REQUEST_TIMEOUT = 2.0

ACTION_COLOURS = {
    "reject":   "#e74c3c",
    "rollback": "#e67e22",
    "retrain":  "#f39c12",
    "promote":  "#2ecc71",
    "none":     "#95a5a6",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_get(url: str) -> Any | None:
    """GET a JSON endpoint; return parsed body or None on any failure."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API unavailable: {url}")
        st.caption(str(exc))
        return None


def coerce_timestamp(series: pd.Series) -> pd.Series:
    """Normalise timestamps that may be ISO strings or UNIX epoch floats."""
    as_unix = pd.to_datetime(series, errors="coerce", unit="s")
    as_iso  = pd.to_datetime(series, errors="coerce")
    return as_unix.fillna(as_iso)


def trust_colour(score: float) -> str:
    """Map trust score to a status colour."""
    if score >= 0.80:
        return "normal"
    if score >= 0.65:
        return "off"
    return "inverse"


def action_badge(action: str) -> str:
    colour = ACTION_COLOURS.get(action, "#95a5a6")
    return f'<span style="background:{colour};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8em">{action}</span>'


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Model Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📊 Model Monitor")
st.caption("Drift-aware, hysteresis-protected model lifecycle automation")

# Auto-refresh control
with st.sidebar:
    st.header("Settings")
    refresh_secs = st.selectbox("Auto-refresh", [0, 10, 30, 60], index=0,
                                format_func=lambda x: "Off" if x == 0 else f"Every {x}s")
    if refresh_secs:
        st.rerun()  # triggers on every render cycle; Streamlit throttles it

# ---------------------------------------------------------------------------
# Service status row
# ---------------------------------------------------------------------------

st.subheader("Service Status")
col_health, col_ready, col_version = st.columns(3)

with col_health:
    health = safe_get(f"{API_BASE}/health")
    if health:
        st.success("✓ Service healthy")
        st.json(health, expanded=False)
    else:
        st.error("✗ Health check failed")

with col_ready:
    ready = safe_get(f"{API_BASE}/ready")
    if ready:
        if ready.get("ready"):
            st.success("✓ Model loaded")
        else:
            st.warning(f"⚠ Not ready: {ready.get('reason', 'unknown')}")
        st.json(ready, expanded=False)
    else:
        st.error("✗ Readiness check failed")

with col_version:
    st.info("ℹ Run `make train` then `make run` to start the server")

# ---------------------------------------------------------------------------
# Live metrics -  colour-coded, with deltas
# ---------------------------------------------------------------------------

st.subheader("Latest Batch Metrics")

latest = safe_get(f"{DASHBOARD}/metrics/latest")

if isinstance(latest, dict) and "accuracy" in latest:
    trust = latest.get("trust_score") or latest.get("avg_trust_score", 0.0)
    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Accuracy",       f"{latest['accuracy']:.4f}")
    m2.metric("F1 Score",       f"{latest['f1']:.4f}")
    m3.metric("Avg Confidence", f"{latest['avg_confidence']:.4f}")
    m4.metric("Drift (PSI)",    f"{latest['drift_score']:.4f}",
              delta=None,
              delta_color="inverse")   # high drift is bad
    m5.metric("Latency",        f"{latest['decision_latency_ms']:.0f} ms")
else:
    st.info("No metrics yet. Run `make train` then `make sim` to populate.")

# ---------------------------------------------------------------------------
# Rolling window trust scores -  colour-coded
# ---------------------------------------------------------------------------

st.subheader("Rolling Window Trust Scores")

window_cols = st.columns(3)
for col, window in zip(window_cols, ["5m", "1h", "24h"]):
    summary = safe_get(f"{DASHBOARD}/metrics/summary/{window}")
    with col:
        st.markdown(f"**{window} window**")
        if isinstance(summary, dict) and summary.get("n_batches", 0) > 0:
            ts = float(summary.get("trust_score") or 0.0)
            f1 = summary.get("avg_f1")
            dr = summary.get("avg_drift_score")
            nb = summary.get("n_batches", 0)

            # Colour-coded trust score
            colour = "#2ecc71" if ts >= 0.80 else "#f39c12" if ts >= 0.65 else "#e74c3c"
            st.markdown(
                f'<p style="font-size:2.2rem;font-weight:700;color:{colour};margin:0">'
                f'{ts:.3f}</p><p style="color:#888;margin:0;font-size:0.85em">'
                f'trust score · {nb} batches</p>',
                unsafe_allow_html=True,
            )
            if f1 is not None:
                st.metric("Avg F1",    f"{f1:.4f}", label_visibility="visible")
            if dr is not None:
                st.metric("Avg Drift", f"{dr:.4f}", delta_color="inverse",
                          label_visibility="visible")
        else:
            st.caption("No data yet")

# ---------------------------------------------------------------------------
# Metrics history charts
# ---------------------------------------------------------------------------

st.subheader("Metrics History")

history = safe_get(f"{DASHBOARD}/metrics/tail?limit=500")

if isinstance(history, list) and history:
    df = pd.DataFrame(history)
    df["timestamp"] = coerce_timestamp(df["timestamp"])
    df = df.sort_values("timestamp")

    tab_perf, tab_drift, tab_latency = st.tabs(["Performance", "Drift", "Latency"])

    with tab_perf:
        st.line_chart(df.set_index("timestamp")[["accuracy", "f1"]],
                      use_container_width=True)

    with tab_drift:
        st.line_chart(df.set_index("timestamp")[["drift_score"]],
                      use_container_width=True)

    with tab_latency:
        if "decision_latency_ms" in df.columns:
            st.line_chart(df.set_index("timestamp")[["decision_latency_ms"]],
                          use_container_width=True)
else:
    st.info("Metrics history unavailable.")

# ---------------------------------------------------------------------------
# Decision history -  with human-readable timestamps and action badges
# ---------------------------------------------------------------------------

st.subheader("Decision History")

decisions = safe_get(f"{DASHBOARD}/decisions/history?limit=200")

if isinstance(decisions, list) and decisions:
    ddf = pd.DataFrame(decisions)
    if "timestamp" in ddf.columns:
        ddf["timestamp"] = coerce_timestamp(ddf["timestamp"])
        ddf["time"] = ddf["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        ddf = ddf.sort_values("timestamp", ascending=False)

    # Summary action counts
    if "action" in ddf.columns:
        counts = ddf["action"].value_counts()
        count_cols = st.columns(len(counts))
        for col, (action, n) in zip(count_cols, counts.items()):
            colour = ACTION_COLOURS.get(str(action), "#95a5a6")
            col.markdown(
                f'<p style="text-align:center;font-size:1.8rem;font-weight:700;'
                f'color:{colour}">{n}</p>'
                f'<p style="text-align:center;color:#888;font-size:0.8em">{action}</p>',
                unsafe_allow_html=True,
            )

    # Table -  show readable columns
    display_cols = [c for c in ["time", "action", "reason", "trust_score", "f1",
                                "drift_score", "model_version"] if c in ddf.columns]
    st.dataframe(ddf[display_cols], use_container_width=True, hide_index=True)

    # Recent decisions explained
    st.markdown("#### Last 5 decisions explained")
    for payload in ddf.head(5).to_dict(orient="records"):
        try:
            decision = decision_from_api(payload)
            exp = format_decision_explanation(decision)
            action = str(payload.get("action", "none"))
            colour = ACTION_COLOURS.get(action, "#95a5a6")
            st.markdown(
                f'<div style="border-left:3px solid {colour};padding:8px 12px;'
                f'margin:6px 0;background:#1a1a1a10;border-radius:0 4px 4px 0">'
                f'<strong>{exp["title"]}</strong> -  {exp["reason"]}<br>'
                f'<small style="color:#888">{exp["details"]}</small></div>',
                unsafe_allow_html=True,
            )
        except (KeyError, TypeError):
            continue
else:
    st.info("No decisions recorded yet. Run `make sim` to generate activity.")

# ---------------------------------------------------------------------------
# Decision simulation
# ---------------------------------------------------------------------------

st.subheader("Decision Simulation")
st.caption(
    "Runs the decision engine against current metric summaries -  "
    "no side effects, no state changes."
)

if st.button("▶  Simulate next decision", type="primary"):
    try:
        resp = requests.post(f"{DASHBOARD}/decisions/simulate", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()
        action = result.get("action", "unknown")
        colour = ACTION_COLOURS.get(action, "#95a5a6")
        st.markdown(
            f'<div style="border:1px solid {colour};padding:12px 16px;border-radius:6px">'
            f'<span style="font-size:1.3rem;font-weight:700;color:{colour}">'
            f'{action.upper()}</span> -  {result.get("reason", "")}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Full response"):
            st.json(result)
    except requests.RequestException as exc:
        st.error("Simulation request failed")
        st.caption(str(exc))

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption("model_monitor · [github.com/bonnie-mcconnell/model_monitor](https://github.com/bonnie-mcconnell/model_monitor)")
