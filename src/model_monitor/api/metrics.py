"""Prometheus /metrics endpoint.

Exports the current system state using the official ``prometheus-client``
library so model_monitor integrates correctly with any Prometheus-compatible
observability stack (Grafana, Datadog Agent, OpenTelemetry Collector, etc.).

Exported metrics
----------------
Gauges (per aggregation window):
    model_monitor_trust_score{window="5m|1h|24h"}
    model_monitor_drift_score{window="..."}          -- input feature PSI
    model_monitor_output_drift_score{window="..."}   -- output distribution PSI
    model_monitor_data_quality_score{window="..."}   -- [0,1] data quality aggregate
    model_monitor_f1{window="..."}
    model_monitor_accuracy{window="..."}
    model_monitor_n_batches{window="..."}
    model_monitor_calibration_error{window="..."}
    model_monitor_conformal_coverage{window="..."}   -- fraction of labeled samples covered
    model_monitor_conformal_set_size{window="..."}   -- mean prediction set size

Counters:
    model_monitor_decisions_total{action="none|retrain|rollback|reject|promote"}

Info / label carriers:
    model_monitor_model_version_info{version="..."}

Histograms:
    model_monitor_decision_latency_ms_bucket{le="...",...}
    (records p50/p95/p99 latency across rolling 5-minute windows)

Scalar gauges:
    model_monitor_f1_baseline   - F1 written at last promotion; horizontal
                                   reference line in Grafana panels

Scrape config
-------------
Add to prometheus.yml::

    scrape_configs:
      - job_name: model_monitor
        static_configs:
          - targets: ['localhost:8000']
        metrics_path: /metrics

The endpoint is unauthenticated; protect it at the network layer in
production (reverse proxy, firewall rule, or Prometheus ``basic_auth``).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.model_store import ModelStore

log = logging.getLogger(__name__)

router = APIRouter(tags=["observability"])

# ---------------------------------------------------------------------------
# Prometheus metric objects
#
# Each metric is a module-level singleton.  prometheus-client uses a global
# default registry; we pass it explicitly so tests can use an isolated
# CollectorRegistry without affecting the module-level instances.
# ---------------------------------------------------------------------------

_REGISTRY = CollectorRegistry()

_trust_score = Gauge(
    "model_monitor_trust_score",
    "Weighted trust score in [0, 1]; lower means the model needs attention.",
    ["window"],
    registry=_REGISTRY,
)

_drift_score = Gauge(
    "model_monitor_drift_score",
    "Mean PSI across all features. Values above 0.2 indicate significant drift.",
    ["window"],
    registry=_REGISTRY,
)

_f1 = Gauge(
    "model_monitor_f1",
    "Weighted-average F1 score in the aggregation window.",
    ["window"],
    registry=_REGISTRY,
)

_accuracy = Gauge(
    "model_monitor_accuracy",
    "Weighted-average accuracy in the aggregation window.",
    ["window"],
    registry=_REGISTRY,
)

_n_batches = Gauge(
    "model_monitor_n_batches",
    "Number of inference batches recorded in the aggregation window.",
    ["window"],
    registry=_REGISTRY,
)

_calibration_error = Gauge(
    "model_monitor_calibration_error",
    "Expected Calibration Error (ECE) in [0, 1]; lower is better.",
    ["window"],
    registry=_REGISTRY,
)

_output_drift_score = Gauge(
    "model_monitor_output_drift_score",
    "Mean PSI across output probability classes (output distribution drift).",
    ["window"],
    registry=_REGISTRY,
)

_data_quality_score = Gauge(
    "model_monitor_data_quality_score",
    "Data quality aggregate score [0, 1]. 1.0 = no null/range/schema issues detected.",
    ["window"],
    registry=_REGISTRY,
)

_conformal_coverage = Gauge(
    "model_monitor_conformal_coverage",
    "Fraction of labeled samples whose true label fell within the conformal prediction set. "
    "Should stay >= (1 - alpha); dropping below signals provable model degradation.",
    ["window"],
    registry=_REGISTRY,
)

_conformal_set_size = Gauge(
    "model_monitor_conformal_set_size",
    "Mean conformal prediction set size. Growing set size signals increasing model uncertainty.",
    ["window"],
    registry=_REGISTRY,
)

_mmd_p_value = Gauge(
    "model_monitor_mmd_p_value",
    "MMD permutation test p-value for joint distribution shift. "
    "Values below alpha (default 0.05) indicate statistically significant joint drift - "
    "the failure mode PSI cannot detect (e.g. correlation structure changes).",
    ["window"],
    registry=_REGISTRY,
)

_mmd_is_drift = Gauge(
    "model_monitor_mmd_is_drift",
    "1.0 when the MMD test rejects the null hypothesis of no joint drift, 0.0 otherwise. "
    "Alert on this gauge to catch distribution shift invisible to per-feature PSI.",
    ["window"],
    registry=_REGISTRY,
)

_regression_mae = Gauge(
    "model_monitor_regression_mae",
    "Mean absolute error of a regression model over the current window.",
    ["window"],
    registry=_REGISTRY,
)

_regression_rmse = Gauge(
    "model_monitor_regression_rmse",
    "Root mean squared error of a regression model over the current window.",
    ["window"],
    registry=_REGISTRY,
)

_regression_wasserstein = Gauge(
    "model_monitor_regression_wasserstein",
    "Wasserstein-1 distance between reference and current prediction distributions. "
    "Detects output distribution drift for regression models.",
    ["window"],
    registry=_REGISTRY,
)


_decisions_total = Counter(
    "model_monitor_decisions_total",
    "Cumulative count of lifecycle decisions by action type.",
    ["action"],
    registry=_REGISTRY,
)

_model_version = Gauge(
    "model_monitor_model_version_info",
    "Label-carrier gauge that records the currently active model version. Value is always 1.",
    ["version"],
    registry=_REGISTRY,
)

_f1_baseline = Gauge(
    "model_monitor_f1_baseline",
    "F1 score written at the most recent model promotion. Horizontal reference in dashboards.",
    registry=_REGISTRY,
)

_latency_histogram = Histogram(
    "model_monitor_decision_latency_ms",
    "End-to-end batch decision latency in milliseconds.",
    ["window"],
    # Buckets cover the latency range that matters: fast (<50ms) through degraded (>1000ms).
    buckets=(10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000),
    registry=_REGISTRY,
)

# ---------------------------------------------------------------------------
# Lazy singletons - same pattern as dashboard.py
# ---------------------------------------------------------------------------

_summary_store: MetricsSummaryStore | None = None
_decision_store: DecisionStore | None = None
_model_store: ModelStore | None = None
_metrics_store: MetricsStore | None = None


def _get_summary_store() -> MetricsSummaryStore:
    global _summary_store
    if _summary_store is None:
        _summary_store = MetricsSummaryStore()
    return _summary_store


def _get_decision_store() -> DecisionStore:
    global _decision_store
    if _decision_store is None:
        _decision_store = DecisionStore()
    return _decision_store


def _get_model_store() -> ModelStore:
    global _model_store
    if _model_store is None:
        _model_store = ModelStore()
    return _model_store


def _get_metrics_store() -> MetricsStore:
    global _metrics_store
    if _metrics_store is None:
        _metrics_store = MetricsStore()
    return _metrics_store


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Track the last-known decision counts so _decisions_total stays a true counter.
# prometheus-client Counter only exposes `inc()`; we store the previous SQL total
# and increment by the delta to avoid double-counting across scrapes.
_last_decision_counts: dict[str, int] = {}

# Track the last model version so we can clear stale label sets on promotion.
_last_model_version: str | None = None


def _update_metrics() -> None:
    """Pull current state from the stores and update all Prometheus metrics.

    Called on every scrape request.  All operations are read-only against the
    database; no writes occur here.  Failures are logged and swallowed so a
    single slow or broken store never causes the scrape to fail - Prometheus
    will mark the target as stale after repeated failures through its own
    mechanisms.
    """
    global _last_model_version

    # ── Rolling window gauges ─────────────────────────────────────────────
    try:
        store = _get_summary_store()
        for window in ("5m", "1h", "24h"):
            summary = store.get(window)
            if summary is None or summary.n_batches == 0:
                continue
            _trust_score.labels(window=window).set(summary.trust_score)
            _drift_score.labels(window=window).set(summary.avg_drift_score)
            _f1.labels(window=window).set(summary.avg_f1)
            _accuracy.labels(window=window).set(summary.avg_accuracy)
            _n_batches.labels(window=window).set(summary.n_batches)
            if summary.avg_calibration_error is not None:
                _calibration_error.labels(window=window).set(
                    summary.avg_calibration_error
                )
            # New monitoring signals - only exported when data is available.
            if (
                hasattr(summary, "avg_output_drift_score")
                and summary.avg_output_drift_score is not None
            ):
                _output_drift_score.labels(window=window).set(
                    summary.avg_output_drift_score
                )
            if (
                hasattr(summary, "avg_data_quality_score")
                and summary.avg_data_quality_score is not None
            ):
                _data_quality_score.labels(window=window).set(
                    summary.avg_data_quality_score
                )
            if (
                hasattr(summary, "avg_conformal_coverage")
                and summary.avg_conformal_coverage is not None
            ):
                _conformal_coverage.labels(window=window).set(
                    summary.avg_conformal_coverage
                )
            if (
                hasattr(summary, "avg_conformal_set_size")
                and summary.avg_conformal_set_size is not None
            ):
                _conformal_set_size.labels(window=window).set(
                    summary.avg_conformal_set_size
                )
            if (
                hasattr(summary, "mmd_p_value")
                and summary.mmd_p_value is not None
            ):
                _mmd_p_value.labels(window=window).set(summary.mmd_p_value)
            if (
                hasattr(summary, "mmd_is_drift")
                and summary.mmd_is_drift is not None
            ):
                _mmd_is_drift.labels(window=window).set(
                    1.0 if summary.mmd_is_drift else 0.0
                )
    except Exception:
        log.exception("metrics_summary_read_failed")

    # ── Decision counters ─────────────────────────────────────────────────
    try:
        ds = _get_decision_store()
        current_counts = ds.count_by_action()
        for action, total in current_counts.items():
            prev = _last_decision_counts.get(action, 0)
            delta = total - prev
            if delta > 0:
                _decisions_total.labels(action=action).inc(delta)
            _last_decision_counts[action] = total
    except Exception:
        log.exception("metrics_decision_count_failed")

    # ── Model version info ────────────────────────────────────────────────
    try:
        ms = _get_model_store()
        version = ms.get_active_version() or "unknown"

        # Clear the old label set when the model is promoted so the Grafana
        # version annotation track shows only the current version with value=1
        # and previous versions disappear rather than staying at stale 1.0.
        if version != _last_model_version and _last_model_version is not None:
            _model_version.remove(_last_model_version)
        _model_version.labels(version=version).set(1.0)
        _last_model_version = version

        # F1 baseline - horizontal reference line in Grafana panels.
        # Explicitly set to NaN when no model is active so the gauge value
        # is deterministic: dashboards show no data point rather than a stale
        # value from the last promoted model in the same process lifetime.
        baseline_f1 = ms.get_active_metadata().get("metrics", {}).get("baseline_f1")
        _f1_baseline.set(
            float(baseline_f1) if baseline_f1 is not None else float("nan")
        )
    except Exception:
        log.exception("metrics_model_version_failed")

    # ── Latency histogram ─────────────────────────────────────────────────
    # Observe recent (last 500) batch latencies into per-window histograms.
    # The histogram is cumulative across the process lifetime; we observe new
    # records since the last scrape to avoid double-counting.
    # This is best-effort; histogram skew on missed records is acceptable.
    try:
        mstore = _get_metrics_store()
        recent = mstore.tail(limit=500)
        for record in recent:
            latency = record.get("decision_latency_ms")
            if latency is not None:
                # Records don't carry a window tag - use "batch" to distinguish
                # from the rolling-window gauges above.
                _latency_histogram.labels(window="batch").observe(float(latency))
    except Exception:
        log.exception("metrics_latency_histogram_failed")


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get(
    "/metrics",
    response_class=Response,
    summary="Prometheus metrics endpoint",
    description=(
        "Returns current monitoring state in Prometheus text exposition format. "
        "Scrape this from your Prometheus instance or Grafana Agent. "
        "The endpoint uses the official prometheus-client library so all metric "
        "types (Counter, Gauge, Histogram) are correct and interoperable with "
        "any Prometheus-compatible backend."
    ),
)
def prometheus_metrics() -> Response:
    """Scrape endpoint: update all metrics then return the current registry."""
    _update_metrics()
    data = generate_latest(_REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
