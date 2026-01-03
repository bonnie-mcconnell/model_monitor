from __future__ import annotations

import asyncio
import time
from typing import Sequence

from model_monitor.monitoring.types import MetricRecord
from model_monitor.monitoring.trust_score import compute_trust_score
from model_monitor.monitoring.alerting import check_alerts
from model_monitor.monitoring.retrain_buffer import RetrainBuffer

from model_monitor.training.retrain_pipeline import RetrainPipeline

from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore


# ---------------------------------------------------------------------
# Aggregation windows
# ---------------------------------------------------------------------

AGGREGATION_WINDOWS: dict[str, int] = {
    "5m": 5 * 60,
    "1h": 60 * 60,
    "24h": 24 * 60 * 60,
}


# ---------------------------------------------------------------------
# Global retrain state (intentionally process-wide)
# ---------------------------------------------------------------------

_retrain_buffer = RetrainBuffer(min_samples=5_000)
_retrain_pipeline = RetrainPipeline()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def aggregate_once(
    metrics_store: MetricsStore,
    summary_store: MetricsSummaryStore,
    *,
    now: float | None = None,
) -> None:
    """
    Run a single aggregation pass.
    Safe for tests, cron jobs, and async loops.
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
        summary["window"] = window
        summary_store.upsert(summary)

        # --- Monitoring actions ---
        check_alerts(window, summary)
        _maybe_trigger_retrain(summary)


async def start_aggregation_loop(
    poll_interval: int = 60,
) -> None:
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
        "n_records": n,
        "accuracy": avg_accuracy,
        "f1": avg_f1,
        "avg_confidence": avg_confidence,
        "drift_score": avg_drift,
        "decision_latency_ms": avg_latency,
        "trust_score": trust_score,
        "trust_components": trust_components,
        "computed_at": time.time(),
    }


def _maybe_trigger_retrain(summary: dict) -> None:
    """
    Decide whether to retrain based on aggregated signals.
    """
    trust = summary["trust_score"]

    # Simple, explainable trigger (can evolve later)
    if trust > 0.65:
        return

    if not _retrain_buffer.ready():
        return

    retrain_df = _retrain_buffer.consume()

    _retrain_pipeline.run(
        retrain_df,
        min_f1_improvement=0.01,
    )
