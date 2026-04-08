"""Aggregation loop: rolls up metric windows, scores trust, fires decisions."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, cast

from model_monitor.config.settings import AppConfig, load_config
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decision_executor import DecisionExecutor
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decisions import DecisionType
from model_monitor.core.default_model_action_executor import DefaultModelActionExecutor
from model_monitor.monitoring.alerting import check_alerts
from model_monitor.monitoring.invariants import (
    assert_bounded,
    assert_monotonic,
    validate_trust_components,
)
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.monitoring.trust_score import (
    TrustScoreComponents,
    compute_trust_score,
)
from model_monitor.monitoring.types import MetricRecord
from model_monitor.monitoring.windows import AGGREGATION_WINDOWS
from model_monitor.storage.behavioral_decision_store import BehavioralDecisionStore
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.model_store import ModelStore
from model_monitor.storage.snapshot_store import SnapshotStore
from model_monitor.training.retrain_pipeline import RetrainPipeline

log = logging.getLogger(__name__)


def _schedule_execution(
    coro: Coroutine[Any, Any, None],
    *,
    window: str,
    action: str,
) -> asyncio.Task[None]:
    """
    Schedule a decision executor coroutine and attach an exception-logging
    callback so failures are never silently swallowed.

    asyncio.create_task fires the coroutine concurrently with the rest of the
    aggregation pass. Without a done-callback, any exception raised inside the
    task is stored on the Task object but never surfaced - Python only prints
    a "Task exception was never retrieved" warning at GC time, which may arrive
    long after the relevant log context is gone.
    """
    task: asyncio.Task[None] = asyncio.create_task(coro)

    def _on_done(t: asyncio.Task[None]) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            log.error(
                "decision_executor_task_failed",
                exc_info=exc,
                extra={"window": window, "action": action},
            )

    task.add_done_callback(_on_done)
    return task


@dataclass(frozen=True)
class AggregatedSummary:
    window: str
    n_batches: int
    avg_accuracy: float
    avg_f1: float
    avg_confidence: float
    avg_drift_score: float
    avg_latency_ms: float
    behavioral_violation_rate: float
    trust_score: float
    trust_components: TrustScoreComponents
    computed_at: float


async def aggregate_once(
    *,
    metrics_store: MetricsStore,
    summary_store: MetricsSummaryStore,
    history_store: MetricsSummaryHistoryStore,
    retrain_buffer: RetrainEvidenceBuffer,
    decision_engine: DecisionEngine,
    decision_executor: DecisionExecutor,
    decision_store: DecisionStore,
    model_store: ModelStore,
    cfg: AppConfig,
    behavioral_store: BehavioralDecisionStore | None = None,
    snapshot_store: SnapshotStore | None = None,
    now: float | None = None,
) -> None:
    """
    Run one aggregation pass across all time windows.

    behavioral_store is optional so callers that have not integrated
    behavioral monitoring (e.g. the main branch, older tests) receive
    a zero behavioral_violation_rate rather than a crash.
    """
    now = now or time.time()

    active_meta = model_store.get_active_metadata()
    baseline_f1: float | None = active_meta.get("metrics", {}).get("baseline_f1")

    for window, seconds in AGGREGATION_WINDOWS.items():
        records, _ = metrics_store.list(limit=10_000, start_ts=now - seconds)
        if not records:
            continue

        # Compute behavioral violation rate over the same time window as
        # the performance metrics so both signals share a consistent denominator.
        bvr = 0.0
        if behavioral_store is not None:
            bvr = behavioral_store.violation_rate(since_ts=now - seconds)

        summary = _aggregate_records(window, records, behavioral_violation_rate=bvr)

        assert_monotonic("n_batches", summary.n_batches)
        assert_bounded("avg_accuracy", summary.avg_accuracy, lo=0.0, hi=1.0)
        assert_bounded("avg_f1", summary.avg_f1, lo=0.0, hi=1.0)
        assert_bounded("avg_confidence", summary.avg_confidence, lo=0.0, hi=1.0)
        assert_bounded("avg_drift_score", summary.avg_drift_score, lo=0.0, hi=1.0)
        validate_trust_components(cast(dict[str, float], summary.trust_components))

        retrain_buffer.add_summary(
            accuracy=summary.avg_accuracy,
            f1=summary.avg_f1,
            drift_score=summary.avg_drift_score,
            trust_score=summary.trust_score,
            timestamp=summary.computed_at,
        )

        summary_store.upsert(
            window=window,
            n_batches=summary.n_batches,
            avg_accuracy=summary.avg_accuracy,
            avg_f1=summary.avg_f1,
            avg_confidence=summary.avg_confidence,
            avg_drift_score=summary.avg_drift_score,
            avg_latency_ms=summary.avg_latency_ms,
            trust_score=summary.trust_score,
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

        effective_baseline = baseline_f1 if baseline_f1 is not None else summary.avg_f1

        recent_raw = decision_store.tail(limit=cfg.retrain.cooldown_batches + 5)
        recent_actions: list[DecisionType] = cast(
            list[DecisionType],
            [r.action for r in recent_raw],
        )

        decision = decision_engine.decide(
            batch_index=summary.n_batches,
            trust_score=summary.trust_score,
            f1=summary.avg_f1,
            f1_baseline=effective_baseline,
            drift_score=summary.avg_drift_score,
            recent_actions=recent_actions,
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
                "baseline_f1": effective_baseline,
                "behavioral_violation_rate": bvr,
            },
        )

        decision_store.record(
            decision=decision,
            batch_index=summary.n_batches,
            trust_score=summary.trust_score,
            f1=summary.avg_f1,
            drift_score=summary.avg_drift_score,
        )

        _schedule_execution(
            decision_executor.execute(
                decision=decision,
                snapshot=snapshot,
                context={"window": window, "n_batches": summary.n_batches},
            ),
            window=window,
            action=decision.action,
        )

        check_alerts(window, {"trust_score": summary.trust_score})


async def start_aggregation_loop(
    *,
    metrics_store: MetricsStore,
    summary_store: MetricsSummaryStore,
    history_store: MetricsSummaryHistoryStore,
    retrain_buffer: RetrainEvidenceBuffer,
    model_store: ModelStore,
    decision_store: DecisionStore,
    behavioral_store: BehavioralDecisionStore | None = None,
    snapshot_store: SnapshotStore | None = None,
    poll_interval: int = 60,
) -> None:
    cfg = load_config()
    decision_engine = DecisionEngine(cfg)

    action_executor = DefaultModelActionExecutor(
        model_store=model_store,
        retrain_pipeline=RetrainPipeline(model_store=model_store),
        decision_store=decision_store,
    )

    decision_executor = DecisionExecutor(
        retrain_buffer=retrain_buffer,
        action_executor=action_executor,
        min_f1_improvement=cfg.retrain.min_f1_gain,
        snapshot_store=snapshot_store,
    )

    while True:
        await aggregate_once(
            metrics_store=metrics_store,
            summary_store=summary_store,
            history_store=history_store,
            retrain_buffer=retrain_buffer,
            decision_engine=decision_engine,
            decision_executor=decision_executor,
            decision_store=decision_store,
            model_store=model_store,
            cfg=cfg,
            behavioral_store=behavioral_store,
            snapshot_store=snapshot_store,
        )
        await asyncio.sleep(poll_interval)


def _aggregate_records(
    window: str,
    records: Sequence[MetricRecord],
    behavioral_violation_rate: float = 0.0,
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
        behavioral_violation_rate=behavioral_violation_rate,
    )

    return AggregatedSummary(
        window=window,
        n_batches=n,
        avg_accuracy=avg_accuracy,
        avg_f1=avg_f1,
        avg_confidence=avg_confidence,
        avg_drift_score=avg_drift,
        avg_latency_ms=avg_latency,
        behavioral_violation_rate=behavioral_violation_rate,
        trust_score=trust_score,
        trust_components=trust_components,
        computed_at=time.time(),
    )
