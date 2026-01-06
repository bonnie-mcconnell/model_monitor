from __future__ import annotations

import asyncio
import time
from typing import Sequence

from model_monitor.monitoring.types import MetricRecord
from model_monitor.monitoring.trust_score import compute_trust_score
from model_monitor.monitoring.alerting import check_alerts
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer

from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)


# ---------------------------------------------------------------------
# Aggregation windows (seconds)
# ---------------------------------------------------------------------

AGGREGATION_WINDOWS: dict[str, int] = {
    "5m": 5 * 60,
    "1h": 60 * 60,
    "24h": 24 * 60 * 60,
}


# ---------------------------------------------------------------------
# Process-wide retrain evidence buffer
# ---------------------------------------------------------------------

_retrain_buffer = RetrainEvidenceBuffer(min_samples=5)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def aggregate_once(
    *,
    metrics_store: MetricsStore,
    summary_store: MetricsSummaryStore,
    now: float | None = None,
) -> None:
    """
    Run a single aggregation pass.

    Responsibilities:
    - aggregate batch-level metrics into rolling windows
    - compute trust scores
    - persist aggregated summaries
    - emit alerts (side-effect only)
    - accumulate retrain evidence

    This function does NOT:
    - make operational decisions
    - trigger retraining
    - mutate model state
    """
    if now is None:
        now = time.time()

    for window, seconds in AGGREGATION_WINDOWS.items():
        records, _ = metrics_store.list(
            limit=10_000,
            start_ts=now - seconds,
        )

        if not records:
            continue

        summary = _aggregate_records(records)

        # ---- Accumulate retrain evidence (monitoring-only) ----
        _retrain_buffer.add_summary(
            accuracy=summary["avg_accuracy"],
            f1=summary["avg_f1"],
            drift_score=summary["avg_drift_score"],
            trust_score=summary["trust_score"],
            timestamp=summary["computed_at"],
        )

        # ---- Persist rolling summary ----
        summary_store.upsert(
            window=window,
            n_batches=summary["n_batches"],
            avg_accuracy=summary["avg_accuracy"],
            avg_f1=summary["avg_f1"],
            avg_confidence=summary["avg_confidence"],
            avg_drift_score=summary["avg_drift_score"],
            avg_latency_ms=summary["avg_latency_ms"],
        )
        
        history_store = MetricsSummaryHistoryStore()

        history_store.write(
            window=window,
            timestamp=summary["computed_at"],
            n_batches=summary["n_batches"],
            avg_accuracy=summary["avg_accuracy"],
            avg_f1=summary["avg_f1"],
            avg_confidence=summary["avg_confidence"],
            avg_drift_score=summary["avg_drift_score"],
            avg_latency_ms=summary["avg_latency_ms"],
        )


        # ---- Side effects only ----
        check_alerts(window, summary)


async def start_aggregation_loop(poll_interval: int = 60) -> None:
    """
    Background async aggregation loop.
    """
    metrics_store = MetricsStore()
    summary_store = MetricsSummaryStore()

    while True:
        aggregate_once(
            metrics_store=metrics_store,
            summary_store=summary_store,
        )
        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _aggregate_records(records: Sequence[MetricRecord]) -> dict:
    """
    Aggregate batch-level metrics.

    Semantics:
    - One record == one inference batch
    - All averages are unweighted batch means
    """
    n = len(records)

    avg_accuracy = sum(r["accuracy"] for r in records) / n
    avg_f1 = sum(r["f1"] for r in records) / n
    avg_confidence = sum(r["avg_confidence"] for r in records) / n
    avg_drift = sum(r["drift_score"] for r in records) / n
    avg_latency = sum(r["decision_latency_ms"] for r in records) / n

    trust_score, trust_components = compute_trust_score(
        accuracy=avg_accuracy,
        f1=avg_f1,
        avg_confidence=avg_confidence,
        drift_score=avg_drift,
        decision_latency_ms=avg_latency,
    )

    return {
        # persisted
        "n_batches": n,
        "avg_accuracy": avg_accuracy,
        "avg_f1": avg_f1,
        "avg_confidence": avg_confidence,
        "avg_drift_score": avg_drift,
        "avg_latency_ms": avg_latency,
        # derived (monitoring only)
        "trust_score": trust_score,
        "trust_components": trust_components,
        "computed_at": time.time(),
    }
