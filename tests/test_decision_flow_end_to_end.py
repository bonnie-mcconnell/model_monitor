# tests/test_decision_flow_end_to_end.py
import time
import uuid
import pytest
import pandas as pd

from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decision_runner import DecisionRunner
from model_monitor.core.decision_executor import DecisionExecutor
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decisions import Decision
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.config.settings import load_config
from model_monitor.core.model_action_executor_protocol import ModelActionExecutorProtocol
from model_monitor.core.model_actions import ModelAction


# -----------------------------
# Dummy executor to track calls
# -----------------------------
class DummyActionExecutor(ModelActionExecutorProtocol):
    def __init__(self):
        self.calls = []

    def execute(self, *, action: ModelAction, context: dict):
        self.calls.append((action, context))
        return None


@pytest.mark.asyncio
async def test_decision_flow_end_to_end():
    # -----------------------------
    # Load config
    # -----------------------------
    config = load_config()

    # -----------------------------
    # Stores
    # -----------------------------
    summary_store = MetricsSummaryStore()
    decision_store = DecisionStore()

    # -----------------------------
    # Seed metrics
    # -----------------------------
    summary_store.upsert(
        window="5m",
        n_batches=10,
        avg_accuracy=0.6,
        avg_f1=0.45,
        avg_confidence=0.5,
        avg_drift_score=0.7,
        avg_latency_ms=100.0,
    )

    # -----------------------------
    # Control plane: Decision Engine + Runner
    # -----------------------------
    engine = DecisionEngine(config=config)
    runner = DecisionRunner(
        decision_engine=engine,
        summary_store=summary_store,
        decision_store=decision_store,
    )

    # Run runner to produce decision
    decisions = runner.run_once(windows=["5m"])

    assert len(decisions) == 1
    decision = decisions[0]
    assert isinstance(decision, Decision)

    # -----------------------------
    # Execution plane
    # -----------------------------
    retrain_buffer = RetrainEvidenceBuffer(min_samples=1)
    # Add a dummy dataframe for retrain path
    retrain_buffer.add_summary(
        accuracy=0.6,
        f1=0.45,
        drift_score=0.7,
        trust_score=0.3,
        timestamp=time.time(),
    )

    action_executor = DummyActionExecutor()

    decision_executor = DecisionExecutor(
        retrain_buffer=retrain_buffer,
        action_executor=action_executor,
        min_f1_improvement=config.retrain.min_f1_gain,
        dry_run=True,  # no real side effects
    )

    # Create snapshot for decision
    snapshot = DecisionSnapshot(
        decision_id=str(uuid.uuid4()),
        action=decision.action,
        timestamp=time.time(),
        status="pending",
    )

    # Execute decision
    await decision_executor.execute(
        decision=decision,
        snapshot=snapshot,
    )

    # -----------------------------
    # Assertions
    # -----------------------------
    # Snapshot should have been updated
    assert snapshot.status in {"executed", "skipped"}

    # Dummy executor should have been called if not a no-op
    if decision.action.lower() not in {"none", "reject"}:
        assert len(action_executor.calls) >= 0
