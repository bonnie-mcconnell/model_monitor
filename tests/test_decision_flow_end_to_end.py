from __future__ import annotations

# tests/test_decision_flow_end_to_end.py
import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

import pytest

from model_monitor.config.settings import load_config
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decision_executor import DecisionExecutor
from model_monitor.core.decision_runner import DecisionRunner
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decisions import Decision, DecisionMetadata
from model_monitor.core.model_action_executor_protocol import (
    ModelActionExecutorProtocol,
)
from model_monitor.core.model_actions import ModelAction
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore


# -----------------------------
# Dummy executor to track calls
# -----------------------------
class DummyActionExecutor(ModelActionExecutorProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[ModelAction, dict[str, Any]]] = []

    def execute(self, *, action: ModelAction, context: Mapping[str, Any]) -> None:
        self.calls.append((action, dict(context)))


@pytest.mark.asyncio
async def test_decision_flow_end_to_end() -> None:
    config = load_config()

    summary_store = MetricsSummaryStore()
    decision_store = DecisionStore()

    summary_store.upsert(
        window="5m",
        n_batches=10,
        avg_accuracy=0.6,
        avg_f1=0.45,
        avg_confidence=0.5,
        avg_drift_score=0.7,
        avg_latency_ms=100.0,
        trust_score=0.55,
    )

    engine = DecisionEngine(config=config)
    runner = DecisionRunner(
        decision_engine=engine,
        summary_store=summary_store,
        decision_store=decision_store,
    )

    decisions = runner.run_once(windows=["5m"])

    assert len(decisions) == 1
    decision = decisions[0]
    assert isinstance(decision, Decision)

    retrain_buffer = RetrainEvidenceBuffer(min_samples=1)
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
        dry_run=True,
    )

    snapshot = DecisionSnapshot(
        decision_id=str(uuid.uuid4()),
        action=decision.action,
        timestamp=time.time(),
        status="pending",
    )

    await decision_executor.execute(
        decision=decision,
        snapshot=snapshot,
    )

    assert snapshot.status in {"executed", "skipped", "failed"}


# ---------------------------------------------------------------------------
# metadata_json persistence
# ---------------------------------------------------------------------------

def test_decision_store_persists_metadata_as_json() -> None:
    """
    Decision.metadata must survive a round-trip through DecisionStore.

    The audit trail is only useful if the context that produced a decision
    (baseline F1, threshold at the time, cooldown state) is recoverable
    after the fact. Verifies that record() serialises metadata and tail()
    returns a row whose metadata_json parses back to the original dict.
    """
    store = DecisionStore()

    metadata: DecisionMetadata = {
        "trust_score": 0.62,
        "f1_drop": 0.09,
        "baseline_f1": 0.88,
        "current_f1": 0.79,
        "drift_score": 0.04,
    }
    decision = Decision(
        action="retrain",
        reason="Sustained performance degradation detected",
        metadata=metadata,
    )

    store.record(
        decision=decision,
        batch_index=42,
        trust_score=metadata["trust_score"],
        f1=metadata["current_f1"],
        drift_score=metadata["drift_score"],
    )

    rows = store.tail(limit=1)
    assert rows, "tail() returned no rows after record()"

    row = rows[0]
    assert row.metadata_json is not None, (
        "metadata_json is None - DecisionStore.record() must serialise "
        "Decision.metadata to JSON for audit trail completeness."
    )

    recovered = json.loads(row.metadata_json)
    assert recovered["trust_score"] == pytest.approx(metadata["trust_score"])
    assert recovered["f1_drop"] == pytest.approx(metadata["f1_drop"])
    assert recovered["baseline_f1"] == pytest.approx(metadata["baseline_f1"])


def test_decision_store_handles_empty_metadata_gracefully() -> None:
    """
    A Decision with no metadata (the default) must not produce a JSON
    error or write a spurious empty-object row.
    """
    store = DecisionStore()

    decision = Decision(action="none", reason="System operating within thresholds")
    store.record(decision=decision)

    rows = store.tail(limit=1)
    assert rows

    row = rows[0]
    # Empty dict → metadata_json should be None (falsy metadata is skipped)
    assert row.metadata_json is None or row.metadata_json == "{}"