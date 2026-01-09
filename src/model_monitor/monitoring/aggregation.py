from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Sequence

from model_monitor.monitoring.types import MetricRecord
from model_monitor.monitoring.trust_score import (
    compute_trust_score,
    TrustScoreComponents,
)
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
# Domain model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class AggregatedSummary:
    window: str
    n_batches: int
    avg_accuracy: float
    avg_f1: float
    avg_confidence: float
    avg_drift_score: float
    avg_latency_ms: float
    trust_score: float
    trust_components: TrustScoreComponents
    computed_at: float


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def aggregate_once(
    *,
    metrics_store: MetricsStore,
    summary_store: MetricsSummaryStore,
    history_store: MetricsSummaryHistoryStore,
    retrain_buffer: RetrainEvidenceBuffer,
    now: float | None = None,
) -> None:
    """
    Run a single aggregation pass.

    Responsibilities:
    - aggregate batch-level metrics into rolling windows
    - compute trust scores
    - persist rolling + historical summaries
    - emit alerts (side-effect only)
    - accumulate retrain evidence

    This function intentionally:
    - does NOT trigger retraining
    - does NOT mutate model state
    """
    now = now or time.time()

    for window, seconds in AGGREGATION_WINDOWS.items():
        records, _ = metrics_store.list(
            limit=10_000,
            start_ts=now - seconds,
        )

        if not records:
            continue

        summary = _aggregate_records(window, records)

        # ---- Accumulate retrain evidence ----
        retrain_buffer.add_summary(
            accuracy=summary.avg_accuracy,
            f1=summary.avg_f1,
            drift_score=summary.avg_drift_score,
            trust_score=summary.trust_score,
            timestamp=summary.computed_at,
        )

        # ---- Persist rolling summary ----
        summary_store.upsert(
            window=window,
            n_batches=summary.n_batches,
            avg_accuracy=summary.avg_accuracy,
            avg_f1=summary.avg_f1,
            avg_confidence=summary.avg_confidence,
            avg_drift_score=summary.avg_drift_score,
            avg_latency_ms=summary.avg_latency_ms,
        )

        # ---- Persist historical snapshot ----
        history_store.write(
            window=window,
            timestamp=summary.computed_at,
            n_batches=summary.n_batches,
            avg_accuracy=summary.avg_accuracy,
            avg_f1=summary.avg_f1,
            avg_confidence=summary.avg_confidence,
            avg_drift_score=summary.avg_drift_score,
            avg_latency_ms=summary.avg_latency_ms,
        )

        # ---- Side effects only ----
        check_alerts(window, {
            "trust_score": summary.trust_score,
        })


async def start_aggregation_loop(
    *,
    metrics_store: MetricsStore,
    summary_store: MetricsSummaryStore,
    history_store: MetricsSummaryHistoryStore,
    retrain_buffer: RetrainEvidenceBuffer,
    poll_interval: int = 60,
) -> None:
    """
    Background async aggregation loop.
    """
    while True:
        aggregate_once(
            metrics_store=metrics_store,
            summary_store=summary_store,
            history_store=history_store,
            retrain_buffer=retrain_buffer,
        )
        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _aggregate_records(
    window: str,
    records: Sequence[MetricRecord],
) -> AggregatedSummary:
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

    return AggregatedSummary(
        window=window,
        n_batches=n,
        avg_accuracy=avg_accuracy,
        avg_f1=avg_f1,
        avg_confidence=avg_confidence,
        avg_drift_score=avg_drift,
        avg_latency_ms=avg_latency,
        trust_score=trust_score,
        trust_components=trust_components,
        computed_at=time.time(),
    )
