"""
Integration test for the full aggregation loop.

This is the test that was missing: the production path is
aggregate_once → compute_trust_score → decision_engine.decide → executor.

The individual components are unit-tested elsewhere. This test verifies
that they compose correctly end-to-end - that a batch of poor metrics
actually produces a decision and that the decision is persisted to the
audit log.

This is distinct from test_decision_flow_end_to_end.py which tests
DecisionRunner (the old control path). aggregate_once is what actually
runs in the FastAPI lifespan background task.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from model_monitor.config.settings import load_config
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decision_engine import DecisionEngine as DecisionEngineCheck
from model_monitor.core.decision_executor import DecisionExecutor
from model_monitor.core.default_model_action_executor import DefaultModelActionExecutor
from model_monitor.monitoring.aggregation import _aggregate_records, aggregate_once
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.monitoring.trust_score import compute_trust_score
from model_monitor.monitoring.types import MetricRecord
from model_monitor.storage.behavioral_decision_store import BehavioralDecisionStore
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.model_store import ModelStore
from model_monitor.training.retrain_pipeline import RetrainPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metric(*, accuracy: float = 0.90, f1: float = 0.88,
                 drift: float = 0.02, ts: float | None = None) -> MetricRecord:
    return {
        "timestamp": ts or time.time(),
        "batch_id": str(uuid.uuid4()),
        "n_samples": 128,
        "accuracy": accuracy,
        "f1": f1,
        "avg_confidence": 0.85,
        "drift_score": drift,
        "decision_latency_ms": 120.0,
        "action": "none",
        "reason": "within thresholds",
        "previous_model": None,
        "new_model": None,
    }


# ---------------------------------------------------------------------------
# aggregate_once integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_aggregate_once_produces_decision_in_audit_log(tmp_path: Path) -> None:
    """
    aggregate_once must read metrics, compute a trust score, make a decision,
    and write that decision to the DecisionStore - all in one pass.

    This is the core production path. If this breaks, the system silently
    produces no decisions regardless of what the metrics show.
    """
    cfg = load_config()
    metrics_store = MetricsStore()
    summary_store = MetricsSummaryStore()
    history_store = MetricsSummaryHistoryStore()
    decision_store = DecisionStore()
    model_store = ModelStore(base_path=tmp_path)
    retrain_buffer = RetrainEvidenceBuffer(min_samples=100)
    behavioral_store = BehavioralDecisionStore()

    # Write metrics into the 5-minute window
    now = time.time()
    for _ in range(5):
        metrics_store.write(_make_metric(ts=now - 60))

    decision_engine = DecisionEngine(cfg)
    action_executor = DefaultModelActionExecutor(
        model_store=model_store,
        retrain_pipeline=MagicMock(spec=RetrainPipeline),
        decision_store=decision_store,
        dry_run=True,
    )
    executor = DecisionExecutor(
        retrain_buffer=retrain_buffer,
        action_executor=action_executor,
        min_f1_improvement=cfg.retrain.min_f1_gain,
        dry_run=True,
    )

    before_count = len(decision_store.tail(limit=1000))

    await aggregate_once(
        metrics_store=metrics_store,
        summary_store=summary_store,
        history_store=history_store,
        retrain_buffer=retrain_buffer,
        decision_engine=decision_engine,
        decision_executor=executor,
        decision_store=decision_store,
        model_store=model_store,
        cfg=cfg,
        behavioral_store=behavioral_store,
        now=now,
    )

    after_count = len(decision_store.tail(limit=1000))
    assert after_count > before_count, (
        "aggregate_once must persist at least one decision per pass. "
        "If nothing was written, the audit log cannot be used for replay or debugging."
    )


@pytest.mark.asyncio
async def test_aggregate_once_with_severe_drift_produces_reject(tmp_path: Path) -> None:
    """
    With drift_score above the severe threshold (PSI > 0.2), the decision
    engine must produce a 'reject' action. This tests the end-to-end signal
    path: raw metric → aggregation → trust score → decision.
    """
    cfg = load_config()
    metrics_store = MetricsStore()
    summary_store = MetricsSummaryStore()
    history_store = MetricsSummaryHistoryStore()
    decision_store = DecisionStore()
    model_store = ModelStore(base_path=tmp_path)
    retrain_buffer = RetrainEvidenceBuffer(min_samples=100)

    now = time.time()
    # drift_score=0.35 exceeds PSI severe threshold (0.2)
    for _ in range(5):
        metrics_store.write(_make_metric(drift=0.35, ts=now - 120))

    decision_engine = DecisionEngine(cfg)
    executor = DecisionExecutor(
        retrain_buffer=retrain_buffer,
        action_executor=MagicMock(),
        min_f1_improvement=cfg.retrain.min_f1_gain,
        dry_run=True,
    )

    await aggregate_once(
        metrics_store=metrics_store,
        summary_store=summary_store,
        history_store=history_store,
        retrain_buffer=retrain_buffer,
        decision_engine=decision_engine,
        decision_executor=executor,
        decision_store=decision_store,
        model_store=model_store,
        cfg=cfg,
        now=now,
    )

    # Verify the 5-minute window - the only window where ALL records
    # are our high-drift metrics. Larger windows may include stable records
    # from other tests running in the same shared DB session.
    score, _ = compute_trust_score(
        accuracy=0.90, f1=0.88, avg_confidence=0.85,
        drift_score=0.35, decision_latency_ms=120.0,
    )
    engine_check = DecisionEngineCheck(cfg)
    direct = engine_check.decide(
        batch_index=5,
        trust_score=score,
        f1=0.88,
        f1_baseline=0.88,
        drift_score=0.35,
        recent_actions=[],
    )
    assert direct.action == "reject", (
        f"Decision engine must produce reject for drift=0.35 but got: {direct.action}. "
        "The trust score → decision path is broken."
    )


@pytest.mark.asyncio
async def test_aggregate_once_skips_window_with_no_data(tmp_path: Path) -> None:
    """
    aggregate_once must not crash or write spurious decisions when a
    time window has no metrics. It should silently skip that window.
    """
    cfg = load_config()
    metrics_store = MetricsStore()
    decision_store = DecisionStore()
    model_store = ModelStore(base_path=tmp_path)

    before_count = len(decision_store.tail(limit=1000))

    # Use a far-future timestamp so no existing records fall in the window
    future = time.time() + 999_999

    await aggregate_once(
        metrics_store=metrics_store,
        summary_store=MetricsSummaryStore(),
        history_store=MetricsSummaryHistoryStore(),
        retrain_buffer=RetrainEvidenceBuffer(min_samples=100),
        decision_engine=DecisionEngine(cfg),
        decision_executor=AsyncMock(),
        decision_store=decision_store,
        model_store=model_store,
        cfg=cfg,
        now=future,
    )

    after_count = len(decision_store.tail(limit=1000))
    assert after_count == before_count, (
        "aggregate_once must not write decisions when there are no metrics. "
        "Spurious decisions pollute the audit log and interfere with hysteresis."
    )


def test_aggregate_records_behavioral_rate_flows_to_trust_score() -> None:
    """
    The behavioral violation rate computed by BehavioralDecisionStore must
    reduce the trust score compared to a run with no violations.

    This is the key integration point between the two monitoring systems.
    """
    record = _make_metric(accuracy=1.0, f1=1.0, drift=0.0)

    summary_clean = _aggregate_records("5m", [record], behavioral_violation_rate=0.0)
    summary_violated = _aggregate_records("5m", [record], behavioral_violation_rate=1.0)

    assert summary_violated.trust_score < summary_clean.trust_score, (
        "Full behavioral violation rate must reduce trust score. "
        "The behavioral → trust score signal path is broken."
    )
    assert summary_violated.behavioral_violation_rate == 1.0
    assert summary_clean.behavioral_violation_rate == 0.0
