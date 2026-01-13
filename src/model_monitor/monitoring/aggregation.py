from __future__ import annotations

import asyncio
import time
from typing import cast
import uuid
from dataclasses import dataclass
from typing import Sequence

from model_monitor.monitoring.types import MetricRecord
from model_monitor.monitoring.trust_score import (
    compute_trust_score,
    TrustScoreComponents,
)
from model_monitor.monitoring.alerting import check_alerts
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.monitoring.invariants import (
    assert_bounded,
    assert_monotonic,
    validate_trust_components,
)

from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decision_executor import DecisionExecutor

from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.model_store import ModelStore
from model_monitor.training.retrain_pipeline import RetrainPipeline
from model_monitor.core.model_action_executor import ModelActionExecutor
from model_monitor.config.settings import load_config


AGGREGATION_WINDOWS: dict[str, int] = {
    "5m": 5 * 60,
    "1h": 60 * 60,
    "24h": 24 * 60 * 60,
}


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


def aggregate_once(
    *,
    metrics_store: MetricsStore,
    summary_store: MetricsSummaryStore,
    history_store: MetricsSummaryHistoryStore,
    retrain_buffer: RetrainEvidenceBuffer,
    decision_engine: DecisionEngine,
    decision_executor: DecisionExecutor,
    now: float | None = None,
) -> None:
    now = now or time.time()

    for window, seconds in AGGREGATION_WINDOWS.items():
        records, _ = metrics_store.list(
            limit=10_000,
            start_ts=now - seconds,
        )

        if not records:
            continue

        summary = _aggregate_records(window, records)

        # ---------------------------------
        # Aggregation invariants
        # ---------------------------------
        assert_monotonic("n_batches", summary.n_batches)

        assert_bounded("avg_accuracy", summary.avg_accuracy, lo=0.0, hi=1.0)
        assert_bounded("avg_f1", summary.avg_f1, lo=0.0, hi=1.0)
        assert_bounded("avg_confidence", summary.avg_confidence, lo=0.0, hi=1.0)
        assert_bounded("avg_drift_score", summary.avg_drift_score, lo=0.0, hi=1.0)

        validate_trust_components(
            cast(dict[str, float], summary.trust_components)
        )



        # -------------------------
        # Retrain evidence
        # -------------------------
        retrain_buffer.add_summary(
            accuracy=summary.avg_accuracy,
            f1=summary.avg_f1,
            drift_score=summary.avg_drift_score,
            trust_score=summary.trust_score,
            timestamp=summary.computed_at,
        )

        # -------------------------
        # Persistence
        # -------------------------
        summary_store.upsert(
            window=window,
            n_batches=summary.n_batches,
            avg_accuracy=summary.avg_accuracy,
            avg_f1=summary.avg_f1,
            avg_confidence=summary.avg_confidence,
            avg_drift_score=summary.avg_drift_score,
            avg_latency_ms=summary.avg_latency_ms,
        )

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

        # -------------------------
        # Decision
        # -------------------------
        decision = decision_engine.decide(
            batch_index=summary.n_batches,
            trust_score=summary.trust_score,
            f1=summary.avg_f1,
            f1_baseline=summary.avg_f1,  # TODO: baseline wiring
            drift_score=summary.avg_drift_score,
            recent_actions=None,
        )

        snapshot = DecisionSnapshot(
            decision_id=str(uuid.uuid4()),
            action=decision.action,
            timestamp=summary.computed_at,
            status="pending",
            metadata={
                **decision.metadata,
                "window": window,
                "n_batches": summary.n_batches,
            },
        )

        asyncio.create_task(
            decision_executor.execute(
                decision=decision,
                snapshot=snapshot,
            )
        )

        check_alerts(window, {"trust_score": summary.trust_score})


async def start_aggregation_loop(
    *,
    metrics_store: MetricsStore,
    summary_store: MetricsSummaryStore,
    history_store: MetricsSummaryHistoryStore,
    retrain_buffer: RetrainEvidenceBuffer,
    model_store: ModelStore,
    poll_interval: int = 60,
) -> None:
    cfg = load_config()

    decision_engine = DecisionEngine(cfg)
    decision_store = DecisionStore()

    action_executor = ModelActionExecutor(
        model_store=model_store,
        retrain_pipeline=RetrainPipeline(),
        decision_store=decision_store,
    )

    decision_executor = DecisionExecutor(
        retrain_buffer=retrain_buffer,
        action_executor=action_executor,
        min_f1_improvement=cfg.retrain.min_f1_gain,
    )

    while True:
        aggregate_once(
            metrics_store=metrics_store,
            summary_store=summary_store,
            history_store=history_store,
            retrain_buffer=retrain_buffer,
            decision_engine=decision_engine,
            decision_executor=decision_executor,
        )
        await asyncio.sleep(poll_interval)


def _aggregate_records(
    window: str,
    records: Sequence[MetricRecord],
) -> AggregatedSummary:
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
