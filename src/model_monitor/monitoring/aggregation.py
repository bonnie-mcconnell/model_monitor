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
    assert_non_negative,
    validate_trust_components,
)
from model_monitor.monitoring.raw_data_buffer import RawDataBuffer
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.monitoring.trust_score import (
    TrustScoreComponents,
    compute_trust_score,
)
from model_monitor.monitoring.types import MetricRecord
from model_monitor.monitoring.windows import AGGREGATION_WINDOWS
from model_monitor.storage.alert_store import AlertStore
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.model_store import ModelStore
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
    trust_score: float
    trust_components: TrustScoreComponents
    avg_calibration_error: float | None
    avg_output_drift_score: float | None
    avg_data_quality_score: float | None
    avg_conformal_coverage: float | None
    avg_conformal_set_size: float | None
    computed_at: float
    behavioral_violation_rate: float = 0.0
    # MMD joint-distribution drift statistics aggregated over the window.
    # mmd_p_value: mean p-value of MMD tests run in this window.
    # mmd_is_drift: True when any batch in the window had mmd_is_drift=True.
    mmd_p_value: float | None = None
    mmd_is_drift: bool | None = None


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
    alert_store: AlertStore | None = None,
    behavioral_store: object | None = None,  # BM branch: BehavioralDecisionStore
    snapshot_store: object | None = None,  # BM branch: SnapshotStore
    now: float | None = None,
) -> None:
    now = now or time.time()

    # Read baseline once per aggregation pass, same for all windows
    active_meta = model_store.get_active_metadata()
    baseline_f1: float | None = active_meta.get("metrics", {}).get("baseline_f1")

    for window, seconds in AGGREGATION_WINDOWS.items():
        records, _ = metrics_store.list(limit=10_000, start_ts=now - seconds)
        if not records:
            continue

        summary = _aggregate_records(window, records, cfg)

        assert_non_negative("n_batches", summary.n_batches)
        assert_bounded("avg_accuracy", summary.avg_accuracy, lo=0.0, hi=1.0)
        assert_bounded("avg_f1", summary.avg_f1, lo=0.0, hi=1.0)
        assert_bounded("avg_confidence", summary.avg_confidence, lo=0.0, hi=1.0)
        # PSI is unbounded above - a value of 1.53 is normal under severe drift.
        # Only non-negativity is a hard invariant here.
        assert_non_negative("avg_drift_score", summary.avg_drift_score)
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
            avg_calibration_error=summary.avg_calibration_error,
            avg_output_drift_score=summary.avg_output_drift_score,
            avg_data_quality_score=summary.avg_data_quality_score,
            avg_conformal_coverage=summary.avg_conformal_coverage,
            avg_conformal_set_size=summary.avg_conformal_set_size,
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

        # When no baseline exists yet (first ever deployment), f1_baseline=avg_f1
        # means f1_drop=0, so retrain/rollback are safely suppressed.
        # Drift-based reject still fires normally. This is intentional.
        effective_baseline = baseline_f1 if baseline_f1 is not None else summary.avg_f1

        # Load recent actions for hysteresis - promote requires N stable batches
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
            candidate_exists=model_store.has_candidate(),
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
            },
        )

        # Persist to audit log (single decision path)
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

        check_alerts(
            window, {"trust_score": summary.trust_score}, alert_store=alert_store
        )


async def start_aggregation_loop(
    *,
    metrics_store: MetricsStore,
    summary_store: MetricsSummaryStore,
    history_store: MetricsSummaryHistoryStore,
    retrain_buffer: RetrainEvidenceBuffer,
    model_store: ModelStore,
    decision_store: DecisionStore,
    alert_store: AlertStore | None = None,
    raw_data_buffer: RawDataBuffer | None = None,
    behavioral_store: object | None = None,  # BM branch: BehavioralDecisionStore
    snapshot_store: object | None = None,  # BM branch: SnapshotStore
    poll_interval: int = 60,
    retention_days: int = 30,
) -> None:
    """Start the aggregation and decision loop.

    Args:
        poll_interval:    seconds between aggregation passes.
        retention_days:   delete ``MetricsRecord``s older than this many days
                          on each pass.  Set to 0 to disable pruning.
        raw_data_buffer:  rolling buffer of labeled (X, y) inference pairs.
                          When provided, retrains train on observed data.  When
                          absent, the executor falls back to synthetic data.
    """
    cfg = load_config()
    decision_engine = DecisionEngine(cfg)

    action_executor = DefaultModelActionExecutor(
        model_store=model_store,
        retrain_pipeline=RetrainPipeline(model_store=model_store),
        decision_store=decision_store,
        raw_data_buffer=raw_data_buffer,
    )

    decision_executor = DecisionExecutor(
        retrain_buffer=retrain_buffer,
        action_executor=action_executor,
        min_f1_improvement=cfg.retrain.min_f1_gain,
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
            alert_store=alert_store,
        )
        # Prune old records once per pass to prevent unbounded DB growth.
        if retention_days > 0:
            cutoff = time.time() - retention_days * 86_400
            pruned = metrics_store.prune_before(cutoff)
            if pruned > 0:
                log.info(
                    "metrics_pruned",
                    extra={"n": pruned, "retention_days": retention_days},
                )
        await asyncio.sleep(poll_interval)


def _aggregate_records(
    window: str,
    records: Sequence[MetricRecord],
    cfg: AppConfig | None = None,
    behavioral_violation_rate: float | None = None,
) -> AggregatedSummary:
    n = len(records)
    total_samples = sum(r["n_samples"] for r in records)

    # Accuracy and F1 are percentages over samples - weight by n_samples so a
    # 1000-sample batch and a 10-sample batch are not treated equally.
    # Drift, confidence, and latency are not proportional to sample count so
    # an unweighted mean is appropriate for those.
    if total_samples > 0:
        avg_accuracy = (
            sum(r["accuracy"] * r["n_samples"] for r in records) / total_samples
        )
        avg_f1 = sum(r["f1"] * r["n_samples"] for r in records) / total_samples
    else:
        avg_accuracy = sum(r["accuracy"] for r in records) / n
        avg_f1 = sum(r["f1"] for r in records) / n

    avg_confidence = sum(r["avg_confidence"] for r in records) / n
    avg_drift = sum(r["drift_score"] for r in records) / n
    avg_latency = sum(r["decision_latency_ms"] for r in records) / n

    # ECE: only average over records that have calibration data (non-null)
    ece_values: list[float] = [
        r["calibration_error"]
        for r in records
        if r.get("calibration_error") is not None
        and isinstance(r["calibration_error"], float)
    ]
    avg_ece: float | None = (sum(ece_values) / len(ece_values)) if ece_values else None

    def _mean(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    avg_output_drift = _mean(
        [
            float(r["output_drift_score"])
            for r in records
            if r.get("output_drift_score") is not None
            and isinstance(r["output_drift_score"], (int, float))
        ]
    )
    avg_dq = _mean(
        [
            float(r["data_quality_score"])
            for r in records
            if r.get("data_quality_score") is not None
            and isinstance(r["data_quality_score"], (int, float))
        ]
    )
    avg_conf_cov = _mean(
        [
            float(r["conformal_coverage"])
            for r in records
            if r.get("conformal_coverage") is not None
            and isinstance(r["conformal_coverage"], (int, float))
        ]
    )
    avg_conf_size = _mean(
        [
            float(r["conformal_set_size"])
            for r in records
            if r.get("conformal_set_size") is not None
            and isinstance(r["conformal_set_size"], (int, float))
        ]
    )

    # Behavioral violation rate: use explicit param (for tests/BM wiring) or
    # compute from records as the mean of non-null EMA values in this window.
    avg_bvr: float
    if behavioral_violation_rate is not None:
        avg_bvr = behavioral_violation_rate
    else:
        bvr_vals = [
            float(r["behavioral_violation_rate"])
            for r in records
            if r.get("behavioral_violation_rate") is not None
            and isinstance(r["behavioral_violation_rate"], (int, float))
        ]
        avg_bvr = sum(bvr_vals) / len(bvr_vals) if bvr_vals else 0.0

    trust_score, trust_components = compute_trust_score(
        accuracy=avg_accuracy,
        f1=avg_f1,
        avg_confidence=avg_confidence,
        drift_score=avg_drift,
        decision_latency_ms=avg_latency,
        output_drift_score=avg_output_drift,
        data_quality_score=avg_dq,
        behavioral_violation_rate=avg_bvr,
        config=cfg.trust_score if cfg is not None else None,
    )

    # MMD: mean p-value over batches that ran the test; any-drift flag.
    mmd_pvals: list[float] = [
        float(r["mmd_p_value"])
        for r in records
        if r.get("mmd_p_value") is not None and isinstance(r["mmd_p_value"], (int, float))
    ]
    avg_mmd_p: float | None = (sum(mmd_pvals) / len(mmd_pvals)) if mmd_pvals else None
    any_mmd_drift: bool | None = (
        any(bool(r.get("mmd_is_drift")) for r in records if r.get("mmd_p_value") is not None)
        if mmd_pvals
        else None
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
        avg_calibration_error=avg_ece,
        avg_output_drift_score=avg_output_drift,
        avg_data_quality_score=avg_dq,
        avg_conformal_coverage=avg_conf_cov,
        avg_conformal_set_size=avg_conf_size,
        computed_at=time.time(),
        behavioral_violation_rate=avg_bvr,
        mmd_p_value=avg_mmd_p,
        mmd_is_drift=any_mmd_drift,
    )
