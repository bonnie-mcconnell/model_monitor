"""Streamlit monitoring dashboard - connects to the FastAPI backend."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

try:
    import requests as _requests

    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from model_monitor.ui.decision_explanation import (
    decision_from_api,
    format_decision_explanation,
)

API_BASE = "http://localhost:8000"
DASHBOARD = f"{API_BASE}/dashboard"
REQUEST_TIMEOUT = 3.0

ACTION_COLOURS = {
    "reject": "#e74c3c",
    "rollback": "#e67e22",
    "retrain": "#f39c12",
    "promote": "#2ecc71",
    "none": "#95a5a6",
}
SEVERITY_COLOURS = {"critical": "#e74c3c", "warning": "#f39c12"}


def safe_get(url: str) -> Any | None:
    if not _REQUESTS_AVAILABLE:
        st.error("requests not installed - pip install model-monitor[ui]")
        return None
    try:
        resp = _requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except _requests.RequestException as exc:
        st.error(f"API unavailable: {url}")
        st.caption(str(exc))
        return None


def safe_post(url: str) -> Any | None:
    if not _REQUESTS_AVAILABLE:
        st.error("requests not installed - pip install model-monitor[ui]")
        return None
    try:
        resp = _requests.post(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except _requests.RequestException as exc:
        st.error(f"POST failed: {url}")
        st.caption(str(exc))
        return None


def coerce_timestamp(series: pd.Series) -> pd.Series:
    as_unix = pd.to_datetime(series, errors="coerce", unit="s")
    as_iso = pd.to_datetime(series, errors="coerce")
    return as_unix.fillna(as_iso)


def trust_colour(score: float) -> str:
    if score >= 0.80:
        return "#2ecc71"
    if score >= 0.65:
        return "#f39c12"
    return "#e74c3c"


def action_badge(action: str) -> str:
    colour = ACTION_COLOURS.get(action, "#95a5a6")
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.8em;font-weight:600">{action}</span>'
    )


st.set_page_config(
    page_title="Model Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")

    st.subheader("Auto-refresh")
    refresh_secs = st.selectbox(
        "Interval",
        [0, 10, 30, 60],
        format_func=lambda x: "Off" if x == 0 else f"Every {x}s",
        index=0,
    )

    st.divider()
    st.subheader("Quick actions")
    if st.button("▶ Simulate decision", type="primary", use_container_width=True):
        result = safe_post(f"{DASHBOARD}/decisions/simulate")
        if result:
            action = result.get("action", "unknown")
            colour = ACTION_COLOURS.get(action, "#95a5a6")
            st.markdown(
                f'<div style="border:2px solid {colour};padding:10px;border-radius:6px">'
                f'<strong style="color:{colour}">{action.upper()}</strong><br>'
                f"<small>{result.get('reason', '')}</small><br>"
                f'<small style="color:#888">trust={result.get("trust_score", 0):.3f}</small>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.caption(
        "model_monitor · [GitHub](https://github.com/bonnie-mcconnell/model_monitor)"
    )

# ── Auto-refresh (correct: sleep then rerun, not rerun on every render) ───
if refresh_secs > 0:
    time.sleep(refresh_secs)
    st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📊 Model Monitor")
st.caption("Drift-aware, hysteresis-protected model lifecycle automation")

# ── 1. Service status ──────────────────────────────────────────────────────
st.subheader("Service Status")
c1, c2, c3 = st.columns(3)
with c1:
    health = safe_get(f"{API_BASE}/health")
    st.success("✓ Service healthy") if health else st.error("✗ Health check failed")
with c2:
    ready = safe_get(f"{API_BASE}/ready")
    if ready:
        st.success("✓ Model loaded") if ready.get("ready") else st.warning(
            f"⚠ {ready.get('reason', 'unknown')}"
        )
    else:
        st.error("✗ Readiness failed")
with c3:
    active = safe_get(f"{DASHBOARD}/models/active")
    if active:
        version = active.get("version", "?")
        baseline = active.get("metrics", {}).get("baseline_f1")
        promoted = (active.get("promoted_at_utc") or "")[:16]
        st.info(f"🏷 `{version}`")
        if baseline:
            st.caption(f"Baseline F1: {baseline:.4f} · Promoted: {promoted}")
    else:
        st.caption("No active model - run `make train`")

# ── 2. Latest batch metrics ────────────────────────────────────────────────
st.subheader("Latest Batch Metrics")
latest = safe_get(f"{DASHBOARD}/metrics/latest")
if isinstance(latest, dict) and "accuracy" in latest:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{latest['accuracy']:.4f}")
    m2.metric("F1 Score", f"{latest['f1']:.4f}")
    m3.metric("Avg Confidence", f"{latest['avg_confidence']:.4f}")
    m4.metric("Drift (PSI)", f"{latest['drift_score']:.4f}")
    m5.metric("Latency", f"{latest['decision_latency_ms']:.0f} ms")
    if latest.get("calibration_error") is not None:
        st.caption(
            f"📐 Calibration Error (ECE): **{latest['calibration_error']:.4f}** - 0.0 is perfect, > 0.05 warrants investigation"
        )
else:
    st.info("No metrics yet - run `make train` then `make sim`.")

# ── 3. Rolling window trust scores ─────────────────────────────────────────
st.subheader("Rolling Window Trust Scores")
for col, window in zip(st.columns(3), ["5m", "1h", "24h"]):
    summary = safe_get(f"{DASHBOARD}/metrics/summary/{window}")
    with col:
        st.markdown(f"**{window} window**")
        if isinstance(summary, dict) and summary.get("n_batches", 0) > 0:
            ts = float(summary.get("trust_score") or 0.0)
            colour = trust_colour(ts)
            st.markdown(
                f'<p style="font-size:2.4rem;font-weight:700;color:{colour};margin:0">{ts:.3f}</p>'
                f'<p style="color:#888;margin:0;font-size:0.85em">trust · {summary["n_batches"]} batches</p>',
                unsafe_allow_html=True,
            )
            s1, s2, s3 = st.columns(3)
            s1.metric("Avg F1", f"{summary.get('avg_f1', 0):.4f}")
            s2.metric("Avg Drift", f"{summary.get('avg_drift_score', 0):.4f}")
            if summary.get("avg_calibration_error") is not None:
                s3.metric("Avg ECE", f"{summary['avg_calibration_error']:.4f}")
        else:
            st.caption("No data yet")

# ── 4. Per-feature drift heatmap ────────────────────────────────────────────
st.subheader("Per-Feature Drift Heatmap (PSI)")
st.caption("Green < 0.10 (stable) · Amber 0.10–0.20 (moderate) · Red > 0.20 (severe)")

history_raw = safe_get(f"{DASHBOARD}/metrics/tail?limit=50")
if isinstance(history_raw, list):
    feature_rows = [
        r
        for r in history_raw
        if r.get("feature_drift_scores") and isinstance(r["feature_drift_scores"], list)
    ]
    if feature_rows:
        n_features = len(feature_rows[0]["feature_drift_scores"])
        df_drift = pd.DataFrame(
            [r["feature_drift_scores"] for r in feature_rows],
            columns=[f"f{i}" for i in range(n_features)],
        )

        def _psi_colour(val: float) -> str:
            if val >= 0.2:
                return "background-color:#e74c3c;color:white"
            if val >= 0.1:
                return "background-color:#f39c12;color:white"
            return "background-color:#2ecc71;color:white"

        st.dataframe(
            df_drift.style.map(_psi_colour).format("{:.3f}"),  # type: ignore[arg-type]
            use_container_width=True,
            height=min(400, 30 + len(df_drift) * 35),
        )
    else:
        st.info(
            "Per-feature PSI available after the drift buffer fills (run `make sim`)."
        )
else:
    st.info("No metrics history yet.")

# ── 5. SHAP attribution - prediction-relevant drift ────────────────────────
st.subheader("SHAP Attribution Shift (vs training baseline)")
st.caption(
    "PSI shows *that* a feature drifted. "
    "SHAP shift shows whether the model is using it differently - "
    "positive = model relies on this feature more than at training time."
)

if isinstance(history_raw, list):
    shap_rows = [
        r
        for r in history_raw
        if r.get("shap_attribution") and isinstance(r["shap_attribution"], dict)
    ]
    if shap_rows:
        latest_shap: dict[str, float] = shap_rows[-1]["shap_attribution"]
        shap_df = pd.DataFrame(
            [
                {"Feature": k, "Importance shift": v}
                for k, v in sorted(
                    latest_shap.items(), key=lambda x: abs(x[1]), reverse=True
                )
            ]
        )

        def _shap_colour(val: float) -> str:
            if val > 0.02:
                return "background-color:#e74c3c;color:white"
            if val < -0.02:
                return "background-color:#3498db;color:white"
            return ""

        st.dataframe(
            shap_df.style.map(_shap_colour, subset=["Importance shift"]).format(  # type: ignore[arg-type]
                {"Importance shift": "{:+.4f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"From batch `{shap_rows[-1].get('batch_id', 'unknown')}` "
            f"- only computed when drift PSI > 0.10."
        )
    else:
        st.info(
            "SHAP attribution appears when drift PSI > 0.10 and a "
            "`ShapDriftAttributor` is configured in the Predictor."
        )

# ── 5b. Output drift ──────────────────────────────────────────────────────
st.subheader("Output Drift (Prediction Distribution PSI)")
st.caption(
    "PSI monitors input features. Output drift monitors whether the model's "
    "probability distribution is shifting - often detectable *before* "
    "performance degrades."
)
if isinstance(history_raw, list):
    output_drift_rows = [
        r for r in history_raw if r.get("output_drift_score") is not None
    ]
    if output_drift_rows:
        od_df = pd.DataFrame(output_drift_rows)[["timestamp", "output_drift_score"]]
        od_df["timestamp"] = coerce_timestamp(od_df["timestamp"])
        od_df = od_df.sort_values("timestamp")
        st.line_chart(
            od_df.set_index("timestamp")[["output_drift_score"]],
            use_container_width=True,
        )
        latest_od = output_drift_rows[-1]["output_drift_score"]
        colour = (
            "#e74c3c"
            if latest_od > 0.2
            else ("#f39c12" if latest_od > 0.1 else "#2ecc71")
        )
        st.markdown(
            f'Latest output PSI: <span style="color:{colour};font-weight:700">'
            f"{latest_od:.4f}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Output drift data appears after OutputDriftMonitor is configured.")

# ── 5c. Data quality ───────────────────────────────────────────────────────
st.subheader("Data Quality Score")
st.caption(
    "Aggregates null rate, out-of-range violations, and schema consistency. "
    "1.0 = no issues detected. Drops signal upstream data pipeline failures."
)
if isinstance(history_raw, list):
    dq_rows = [r for r in history_raw if r.get("data_quality_score") is not None]
    if dq_rows:
        dq_df = pd.DataFrame(dq_rows)[["timestamp", "data_quality_score"]]
        dq_df["timestamp"] = coerce_timestamp(dq_df["timestamp"])
        dq_df = dq_df.sort_values("timestamp")
        st.line_chart(
            dq_df.set_index("timestamp")[["data_quality_score"]],
            use_container_width=True,
        )
        latest_dq = dq_rows[-1]["data_quality_score"]
        colour = (
            "#2ecc71"
            if latest_dq >= 0.9
            else ("#f39c12" if latest_dq >= 0.7 else "#e74c3c")
        )
        st.markdown(
            f'Latest quality score: <span style="color:{colour};font-weight:700">'
            f"{latest_dq:.4f}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Data quality data appears after DataQualityMonitor is configured.")

# ── 5d. Conformal coverage ─────────────────────────────────────────────────
st.subheader("Conformal Prediction Coverage")
st.caption(
    "When coverage drops below the configured guarantee (default 90%), "
    "the model's predictions are provably worse than expected on this distribution. "
    "Mean set size > 1.5 signals declining confidence even without labels."
)
if isinstance(history_raw, list):
    conf_rows = [r for r in history_raw if r.get("conformal_set_size") is not None]
    if conf_rows:
        conf_df = pd.DataFrame(conf_rows)
        conf_df["timestamp"] = coerce_timestamp(conf_df["timestamp"])
        conf_df = conf_df.sort_values("timestamp")
        c1, c2 = st.columns(2)
        with c1:
            if (
                "conformal_coverage" in conf_df.columns
                and conf_df["conformal_coverage"].notna().any()
            ):
                st.line_chart(
                    conf_df.dropna(subset=["conformal_coverage"]).set_index(
                        "timestamp"
                    )[["conformal_coverage"]],
                    use_container_width=True,
                )
                st.caption("Coverage rate (target ≥ 0.90)")
        with c2:
            st.line_chart(
                conf_df.set_index("timestamp")[["conformal_set_size"]],
                use_container_width=True,
            )
            st.caption("Mean prediction set size (alarm > 1.5)")
    else:
        st.info("Conformal data appears after ConformalMonitor is calibrated.")

# ── 5e. Tail latency ──────────────────────────────────────────────────────
st.subheader("Tail Latency (p95 / p99)")
st.caption(
    "Average latency can look healthy while p99 is pathological. "
    "These percentiles reflect actual tail user experience."
)
if isinstance(history_raw, list):
    lat_rows = [r for r in history_raw if r.get("p95_latency_ms") is not None]
    if lat_rows:
        lat_df = pd.DataFrame(lat_rows)[
            ["timestamp", "p95_latency_ms", "p99_latency_ms", "decision_latency_ms"]
        ]
        lat_df["timestamp"] = coerce_timestamp(lat_df["timestamp"])
        lat_df = lat_df.sort_values("timestamp").set_index("timestamp")
        st.line_chart(
            lat_df[["decision_latency_ms", "p95_latency_ms", "p99_latency_ms"]],
            use_container_width=True,
        )
    else:
        st.info("Tail latency data requires batches of ≥20 samples.")

# ── 6. Performance + drift + latency + calibration charts ───────────────────
st.subheader("Metrics History")
history = safe_get(f"{DASHBOARD}/metrics/tail?limit=500")
if isinstance(history, list) and history:
    df = pd.DataFrame(history)
    df["timestamp"] = coerce_timestamp(df["timestamp"])
    df = df.sort_values("timestamp")
    tab_perf, tab_drift, tab_lat, tab_cal = st.tabs(
        ["📈 Performance", "🌊 Drift", "⏱ Latency", "🎯 Calibration"]
    )
    with tab_perf:
        st.line_chart(
            df.set_index("timestamp")[["accuracy", "f1"]], use_container_width=True
        )
    with tab_drift:
        st.line_chart(
            df.set_index("timestamp")[["drift_score"]], use_container_width=True
        )
    with tab_lat:
        if "decision_latency_ms" in df.columns:
            st.line_chart(
                df.set_index("timestamp")[["decision_latency_ms"]],
                use_container_width=True,
            )
    with tab_cal:
        ece_col = "calibration_error"
        if ece_col in df.columns and df[ece_col].notna().any():
            st.line_chart(
                df.dropna(subset=[ece_col]).set_index("timestamp")[[ece_col]],
                use_container_width=True,
            )
            st.caption(
                "Expected Calibration Error - 0.0 is perfect, > 0.05 warrants investigation."
            )
        else:
            st.info("Calibration data not yet available.")
else:
    st.info("Run `make sim` to generate data.")

# ── 7. Decision history ─────────────────────────────────────────────────────
st.subheader("Decision History")
decisions = safe_get(f"{DASHBOARD}/decisions/history?limit=200")
if isinstance(decisions, list) and decisions:
    ddf = pd.DataFrame(decisions)
    if "timestamp" in ddf.columns:
        ddf["timestamp"] = coerce_timestamp(ddf["timestamp"])
        ddf["time"] = ddf["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        ddf = ddf.sort_values("timestamp", ascending=False)
    if "action" in ddf.columns:
        counts = ddf["action"].value_counts()
        count_cols = st.columns(min(len(counts), 5))
        for col, (action, n) in zip(count_cols, counts.items()):
            colour = ACTION_COLOURS.get(str(action), "#95a5a6")
            col.markdown(
                f'<div style="text-align:center"><p style="font-size:1.8rem;font-weight:700;color:{colour};margin:0">{n}</p>'
                f'<p style="color:#888;font-size:0.8em;margin:0">{action}</p></div>',
                unsafe_allow_html=True,
            )
    display_cols = [
        c
        for c in ["time", "action", "reason", "trust_score", "f1", "drift_score"]
        if c in ddf.columns
    ]
    st.dataframe(ddf[display_cols], use_container_width=True, hide_index=True)
    st.markdown("#### Last 5 decisions explained")
    for payload in ddf.head(5).to_dict(orient="records"):
        try:
            exp = format_decision_explanation(decision_from_api(payload))
            colour = ACTION_COLOURS.get(
                str(payload.get("action", "none") or "none"), "#95a5a6"
            )
            st.markdown(
                f'<div style="border-left:3px solid {colour};padding:8px 12px;margin:4px 0;border-radius:0 4px 4px 0">'
                f"<strong>{exp['title']}</strong> - {exp['reason']}<br>"
                f'<small style="color:#888">{exp["details"]}</small></div>',
                unsafe_allow_html=True,
            )
        except (KeyError, TypeError):
            continue
else:
    st.info("No decisions yet - run `make sim`.")

# ── 8. Alert history ────────────────────────────────────────────────────────
st.subheader("Alert History")
_, alert_col = st.columns([2, 1])
with alert_col:
    sev_filter = st.selectbox("Filter", ["all", "critical", "warning"], key="alert_sev")

alerts = safe_get(
    f"{DASHBOARD}/alerts/history?limit=100"
    + (f"&severity={sev_filter}" if sev_filter != "all" else "")
)
if isinstance(alerts, dict) and alerts.get("items"):
    items = alerts["items"]
    n_crit = sum(1 for a in items if a.get("severity") == "critical")
    n_warn = sum(1 for a in items if a.get("severity") == "warning")
    al1, al2, al3 = st.columns(3)
    al1.metric("Shown", len(items))
    al2.metric("Critical", n_crit)
    al3.metric("Warning", n_warn)
    adf = pd.DataFrame(items)
    adf["timestamp"] = coerce_timestamp(pd.Series(adf["timestamp"]))
    adf["time"] = adf["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    adf = adf.sort_values("timestamp", ascending=False)
    display = [
        c for c in ["time", "severity", "window", "trust_score"] if c in adf.columns
    ]
    st.dataframe(
        adf[display].style.map(
            lambda v: f"color:{SEVERITY_COLOURS.get(str(v), '#888')};font-weight:600",
            subset=["severity"] if "severity" in adf[display].columns else [],
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No alerts fired - trust score has stayed above the operational floor.")

# ── 9. Model version timeline ───────────────────────────────────────────────
st.subheader("Model Version Timeline")
versions = safe_get(f"{DASHBOARD}/models/versions")
active_v = (safe_get(f"{DASHBOARD}/models/active") or {}).get("version")
if isinstance(versions, list) and versions:
    vdf = pd.DataFrame(versions)
    if "created_at" in vdf.columns:
        vdf["created_at"] = pd.to_datetime(vdf["created_at"], errors="coerce")
        vdf["time"] = vdf["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if active_v and "version" in vdf.columns:
        vdf["status"] = vdf["version"].apply(
            lambda v: "🟢 active" if v == active_v else "archived"
        )
    display = [c for c in ["status", "version", "time"] if c in vdf.columns]
    st.dataframe(vdf[display], use_container_width=True, hide_index=True)
else:
    st.info("No archived versions yet.")

# ── 10. Prometheus preview ───────────────────────────────────────────────────
with st.expander("🔭 Prometheus /metrics preview"):
    st.caption("Add to Prometheus scrape config: `metrics_path: /metrics`")
    if not _REQUESTS_AVAILABLE:
        st.warning("requests not installed - pip install model-monitor[ui]")
    else:
        try:
            resp = _requests.get(f"{API_BASE}/metrics", timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                st.code(resp.text[:3000], language="text")
            else:
                st.warning(f"HTTP {resp.status_code}")
        except Exception as exc:
            st.warning(f"Could not fetch /metrics: {exc}")

# ── 11. Shadow mode comparison ───────────────────────────────────────────────
st.subheader("Shadow Mode: Candidate vs Primary")
shadow = safe_get(f"{DASHBOARD}/shadow")
if isinstance(shadow, dict):
    has_candidate = shadow.get("has_candidate", False)
    n_batches = shadow.get("n_batches", 0)
    rec = shadow.get("recommendation", "no_data")

    rec_colour = {
        "promote_candidate": "#2ecc71",
        "keep_primary": "#e67e22",
        "no_data": "#95a5a6",
        "no_candidate_loaded": "#95a5a6",
    }.get(rec, "#95a5a6")
    rec_label = {
        "promote_candidate": "✅ Promote candidate",
        "keep_primary": "⏳ Keep primary - candidate not ahead",
        "no_data": "ℹ️  No shadow data yet",
        "no_candidate_loaded": "ℹ️  No candidate loaded",
    }.get(rec, rec)

    st.markdown(
        f"<div style='padding:0.5rem 1rem;border-radius:6px;"
        f"background:{rec_colour}22;border-left:4px solid {rec_colour};"
        f"color:{rec_colour};font-weight:600'>{rec_label}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Based on {n_batches} shadow batch(es) · {shadow.get('total_samples', 0):,} samples"
    )

    if n_batches > 0:
        sh1, sh2 = st.columns(2)
        with sh1:
            st.markdown("**Primary**")
            st.metric("Mean F1", f"{shadow['mean_primary_f1']:.4f}")
            st.metric("Mean trust", f"{shadow['mean_primary_trust']:.4f}")
        with sh2:
            st.markdown("**Candidate**")
            f1_delta = shadow["mean_candidate_f1"] - shadow["mean_primary_f1"]
            trust_delta = shadow["mean_candidate_trust"] - shadow["mean_primary_trust"]
            st.metric(
                "Mean F1",
                f"{shadow['mean_candidate_f1']:.4f}",
                delta=f"{f1_delta:+.4f}",
            )
            st.metric(
                "Mean trust",
                f"{shadow['mean_candidate_trust']:.4f}",
                delta=f"{trust_delta:+.4f}",
            )
        st.metric(
            "Agreement rate",
            f"{shadow['mean_agreement_rate']:.1%}",
            help="Fraction of predictions where candidate and primary agree",
        )

    if has_candidate and st.button("Reset shadow stats"):
        safe_post(f"{DASHBOARD}/shadow/reset")
        st.rerun()
else:
    st.info("Shadow stats unavailable - is the API running?")

# ── 12. Causal Drift Attribution ─────────────────────────────────────────────
st.subheader("Causal Drift Attribution")
st.caption(
    "Distinguishes genuine distribution shift from upstream pipeline failures "
    "using Granger causality. Determines whether to retrain or page the data team."
)
causal = safe_get(f"{DASHBOARD}/causal-drift/latest")
_causal_report: dict[str, object] = {}
if isinstance(causal, dict) and causal.get("available"):
    _raw_report = causal.get("report")
    if isinstance(_raw_report, str):
        import json as _json

        try:
            _causal_report = _json.loads(_raw_report)
        except Exception:
            _causal_report = {}
    elif isinstance(_raw_report, dict):
        _causal_report = _raw_report

if _causal_report:
    _dom = _causal_report.get("dominant_cause", "unknown")
    _causal_colours = {
        "genuine_shift": "#f39c12",
        "pipeline_failure": "#e74c3c",
        "mixed": "#e67e22",
        "no_drift": "#2ecc71",
        "unknown": "#95a5a6",
    }
    _causal_icons = {
        "genuine_shift": "📊",
        "pipeline_failure": "🔧",
        "mixed": "⚠️",
        "no_drift": "✅",
        "unknown": "❓",
    }
    _col = _causal_colours.get(str(_dom), "#95a5a6")
    _icon = _causal_icons.get(str(_dom), "❓")
    st.markdown(
        f"<div style='padding:0.75rem 1rem;border-radius:6px;"
        f"background:{_col}22;border-left:4px solid {_col};"
        f"margin-bottom:0.75rem'>"
        f"<span style='color:{_col};font-weight:700;font-size:1.1rem'>"
        f"{_icon} {str(_dom).replace('_', ' ').upper()}</span><br>"
        f"<span style='color:#ccc'>{_causal_report.get('recommendation', '')}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    _feat_results = _causal_report.get("feature_results")
    if isinstance(_feat_results, list) and _feat_results:
        _fdf = pd.DataFrame(_feat_results)
        display_cols = [
            c
            for c in ["feature_name", "psi", "drift_class", "is_root_cause"]
            if c in _fdf.columns
        ]
        st.dataframe(_fdf[display_cols], use_container_width=True, hide_index=True)
    ca1, ca2 = st.columns(2)
    with ca1:
        n_drifted = _causal_report.get("n_drifted_features", 0)
        st.metric("Drifted features", n_drifted)
    with ca2:
        batch_id = _causal_report.get("batch_id", "-")
        st.caption(f"Batch: {batch_id}")
elif isinstance(causal, dict) and not causal.get("available"):
    st.info(
        "No drift events recorded yet - causal analysis requires at least one drifted batch."
    )
else:
    st.info("Causal drift endpoint unavailable - is the API running?")

# ── 13. Threshold Advisor ────────────────────────────────────────────────────
st.subheader("Threshold Advisor")
st.caption(
    "Recommends calibrated PSI and trust-score warn thresholds derived from "
    "this deployment's stable-period variance - eliminating alert fatigue from "
    "hand-tuned defaults."
)
ta = safe_get(f"{DASHBOARD}/threshold-advisor/status")
if isinstance(ta, dict) and ta:
    _ready = ta.get("is_ready", False)
    if _ready and isinstance(ta.get("recommendation"), dict):
        _ta_r = ta["recommendation"]
        ta1, ta2, ta3 = st.columns(3)
        ta1.metric(
            "Global PSI warn",
            f"{_ta_r.get('psi_warn_global', 0):.4f}",
            help="Recommended PSI threshold above which to fire a warning alert",
        )
        ta2.metric(
            "Trust score warn",
            f"{_ta_r.get('trust_warn', 0):.4f}",
            help="Recommended trust score below which to fire a warning alert",
        )
        ta3.metric(
            "Observations",
            ta.get("n_observations", 0),
            help="Number of stable-period batches used to calibrate thresholds",
        )
        per_feat = _ta_r.get("psi_warn_per_feature")
        if isinstance(per_feat, dict) and per_feat:
            st.markdown("**Per-feature PSI warn thresholds**")
            _ta_df = pd.DataFrame(
                [{"feature": k, "psi_warn": v} for k, v in per_feat.items()]
            ).sort_values("psi_warn", ascending=False)
            st.dataframe(_ta_df, use_container_width=True, hide_index=True)
        notes = _ta_r.get("notes")
        if isinstance(notes, list) and notes:
            with st.expander("📝 Advisor notes"):
                for note in notes:
                    st.caption(f"• {note}")
    elif ta.get("available", True):
        n_obs = ta.get("n_observations", 0)
        n_needed = ta.get("min_batches_required", 20)
        st.progress(
            min(n_obs / max(n_needed, 1), 1.0),
            text=f"Collecting stable-period observations: {n_obs}/{n_needed} batches",
        )
        st.caption(
            "Run `make sim` to generate baseline observations. "
            "Recommendations appear once enough stable batches have been recorded."
        )
    else:
        st.info(f"Threshold advisor not configured: {ta.get('reason', 'unknown')}")
elif ta is not None:
    st.info("No threshold advisor data yet - run `make sim`.")
else:
    st.info("Threshold advisor endpoint unavailable - is the API running?")

# ── 14. MMD Joint Distribution Drift ────────────────────────────────────────
st.subheader("MMD Joint Distribution Drift")
st.caption(
    "Kernel Maximum Mean Discrepancy test for joint distribution shift - catches "
    "correlation-structure changes that per-feature PSI cannot detect. "
    "p < 0.05 indicates statistically significant joint drift."
)
mmd_data = safe_get(f"{DASHBOARD}/metrics/tail?limit=50")
if isinstance(mmd_data, list) and mmd_data:
    _mmd_records = [r for r in mmd_data if r.get("mmd_p_value") is not None]
    if _mmd_records:
        _mmd_df = pd.DataFrame(_mmd_records)[
            ["batch_id", "mmd_p_value", "mmd_is_drift"]
        ].copy()
        _mmd_df = _mmd_df.rename(
            columns={
                "mmd_p_value": "p-value",
                "mmd_is_drift": "drift detected",
            }
        )
        _latest_p = float(_mmd_records[0]["mmd_p_value"])
        _latest_drift = bool(_mmd_records[0].get("mmd_is_drift", False))
        m1, m2 = st.columns(2)
        m1.metric(
            "Latest MMD p-value",
            f"{_latest_p:.4f}",
            delta=None,
            help="p < 0.05 → joint distribution shift detected",
        )
        m2.metric(
            "Joint drift",
            "⚠ YES" if _latest_drift else "✅ No",
            help="Based on MMD permutation test at α=0.05",
        )
        st.dataframe(_mmd_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No MMD results yet - MMD runs every batch once enough production data arrives. "
            "Enable `enable_mmd=True` in `MonitorConfig` (default: on)."
        )
else:
    st.info("MMD data unavailable - is the API running?")

# ── 15. Regression Monitor ──────────────────────────────────────────────────
st.subheader("Regression Monitor")
st.caption(
    "For regression models: Wasserstein-1 output drift, MAE/RMSE trust components, "
    "and conformal prediction interval coverage."
)
reg = safe_get(f"{DASHBOARD}/regression/latest")
reg_summary = safe_get(f"{DASHBOARD}/regression/summary")
if isinstance(reg, dict) and reg.get("available"):
    ra, rb, rc = st.columns(3)
    ra.metric(
        "Wasserstein dist.",
        f"{reg.get('wasserstein', 0):.4f}"
        if reg.get("wasserstein") is not None
        else "n/a",
        help="W₁ distance between reference and current prediction distributions",
    )
    rb.metric(
        "MAE",
        f"{reg.get('mae', 0):.4f}" if reg.get("mae") is not None else "n/a",
    )
    rc.metric(
        "RMSE",
        f"{reg.get('rmse', 0):.4f}" if reg.get("rmse") is not None else "n/a",
    )
    if reg.get("coverage_rate") is not None:
        st.metric(
            "Conformal interval coverage",
            f"{float(reg['coverage_rate']):.3f}",
            help="Fraction of samples whose true value fell within the conformal interval",
        )
    if isinstance(reg_summary, dict) and reg_summary.get("available"):
        st.caption(
            f"Batches: {reg_summary.get('n_batches', '?')}  |  "
            f"Mean trust: {reg_summary.get('mean_trust_score', 0):.3f}  |  "
            f"MAE baseline: {reg_summary.get('mae_baseline', '?')}"
        )
elif reg is not None:
    st.info(
        "Regression monitor not configured. "
        "Use `RegressionMonitor` for continuous-output models."
    )
else:
    st.info("Regression endpoint unavailable - is the API running?")
