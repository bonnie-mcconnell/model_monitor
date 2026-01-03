import streamlit as st
from pathlib import Path

from dashboard.plots import (
    load_perf,
    plot_accuracy,
    plot_f1,
    plot_entropy,
    plot_performance,
)

PERF_LOG = Path("metrics/performance.csv")

st.set_page_config(layout="wide")
st.title("Model Monitoring Dashboard")

st.sidebar.header("System Status")

# ---- Load data ----
perf = load_perf(PERF_LOG)

# ---- Sidebar metrics ----
if perf is not None and not perf.empty:
    latest = perf.iloc[-1]

    st.sidebar.metric("Last batch", int(latest["batch_id"]))
    st.sidebar.metric("Accuracy", round(latest["accuracy"], 3))
    st.sidebar.metric("F1", round(latest["f1"], 3))
    st.sidebar.metric("Entropy", round(latest["entropy"], 3))
else:
    st.sidebar.info("No performance data yet")

# ---- Main charts ----
if perf is not None and not perf.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(plot_accuracy(perf))
        st.pyplot(plot_f1(perf))

    with col2:
        st.pyplot(plot_entropy(perf))

else:
    st.warning("No metrics logged yet. Run the simulation to populate data.")

# ---- Summary plot ----
fig = plot_performance(PERF_LOG)
if fig is not None:
    st.subheader("Overall Performance Trend")
    st.pyplot(fig)
