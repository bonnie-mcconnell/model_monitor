from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any

import pytest

from model_monitor.core.decision_executor import DecisionExecutor
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decisions import Decision
from model_monitor.core.model_actions import ModelAction
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.storage.snapshot_store import SnapshotStore


class DummyActionExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[ModelAction, Mapping[str, Any]]] = []

    def execute(
        self,
        *,
        action: ModelAction,
        context: Mapping[str, Any],
    ) -> None:
        self.calls.append((action, context))


@pytest.mark.asyncio
async def test_noop_decision_executes_without_side_effects() -> None:
    buffer = RetrainEvidenceBuffer(min_samples=1)
    executor = DummyActionExecutor()

    decision_executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=executor,
        min_f1_improvement=0.05,
    )

    decision = Decision(action="none", reason="no-op")
    snapshot = DecisionSnapshot(
        decision_id=str(uuid.uuid4()),
        action=decision.action,
        timestamp=time.time(),
        status="pending",
    )

    await decision_executor.execute(decision=decision, snapshot=snapshot)

    assert snapshot.status == "executed"
    assert executor.calls == []


@pytest.mark.asyncio
async def test_retrain_skipped_when_buffer_not_ready() -> None:
    buffer = RetrainEvidenceBuffer(min_samples=2)
    executor = DummyActionExecutor()

    decision_executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=executor,
        min_f1_improvement=0.05,
    )

    decision = Decision(action="retrain", reason="low trust")
    snapshot = DecisionSnapshot(
        decision_id=str(uuid.uuid4()),
        action=decision.action,
        timestamp=time.time(),
        status="pending",
    )

    await decision_executor.execute(decision=decision, snapshot=snapshot)

    assert snapshot.status == "skipped"
    assert executor.calls == []


@pytest.mark.asyncio
async def test_retrain_executes_when_buffer_ready() -> None:
    buffer = RetrainEvidenceBuffer(min_samples=1)
    buffer.add_summary(
        accuracy=0.5, f1=0.4, drift_score=0.6,
        trust_score=0.3, timestamp=time.time(),
    )

    executor = DummyActionExecutor()
    decision_executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=executor,
        min_f1_improvement=0.05,
        dry_run=True,
    )

    decision = Decision(action="retrain", reason="trust degraded")
    snapshot = DecisionSnapshot(
        decision_id=str(uuid.uuid4()),
        action=decision.action,
        timestamp=time.time(),
        status="pending",
    )

    await decision_executor.execute(decision=decision, snapshot=snapshot)

    assert snapshot.status == "executed"


@pytest.mark.asyncio
async def test_retrain_skipped_when_retrain_key_known_to_snapshot_store() -> None:
    """
    Crash-recovery path: when SnapshotStore already has a record for the
    retrain_key that the buffer would produce, the executor must skip.

    Sequence that this test simulates:
    1. A prior run consumed the buffer, computed key K, wrote it to SnapshotStore
    2. The process crashed before execution completed
    3. On restart the same buffer data is present (re-added below)
    4. is_retrain_key_known(K) returns True → execution is skipped

    Without SnapshotStore, step 3 would trigger a duplicate retrain of the
    same evidence window.
    """
    # Use a fixed timestamp so the key is deterministic
    fixed_ts = 1_700_000_000.0

    # Step 1: compute the key the buffer will produce (without consuming it)
    buf_for_key = RetrainEvidenceBuffer(min_samples=1)
    buf_for_key.add_summary(
        accuracy=0.5, f1=0.4, drift_score=0.6, trust_score=0.3,
        timestamp=fixed_ts,
    )
    df = buf_for_key.consume()
    pre_computed_key = buf_for_key.retrain_key(df)

    # Step 2: write that key to SnapshotStore (simulating the pre-crash write-ahead)
    snapshot_store = SnapshotStore()
    prior = DecisionSnapshot(
        decision_id="prior-crashed-" + str(uuid.uuid4()),
        action="retrain",
        timestamp=fixed_ts,
        status="pending",
        retrain_key=pre_computed_key,
    )
    snapshot_store.write(prior)

    # Step 3: new run - same evidence data in the buffer
    buffer = RetrainEvidenceBuffer(min_samples=1)
    buffer.add_summary(
        accuracy=0.5, f1=0.4, drift_score=0.6, trust_score=0.3,
        timestamp=fixed_ts,
    )

    dummy = DummyActionExecutor()
    executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=dummy,
        min_f1_improvement=0.05,
        dry_run=False,
        snapshot_store=snapshot_store,
    )

    decision = Decision(action="retrain", reason="degraded")
    snapshot = DecisionSnapshot(
        decision_id=str(uuid.uuid4()),
        action="retrain",
        timestamp=time.time(),
        status="pending",
    )

    # Step 4: executor detects the known key and skips
    await executor.execute(decision=decision, snapshot=snapshot)

    assert snapshot.status == "skipped", (
        "Executor must skip when retrain_key is already in SnapshotStore. "
        "Crash recovery is broken."
    )
    assert dummy.calls == []


@pytest.mark.asyncio
async def test_promote_decision_writes_snapshot_before_execution() -> None:
    """
    For promote/rollback actions the executor must write the snapshot to
    SnapshotStore BEFORE calling the action executor. If it crashes between
    write and execute, the snapshot remains as evidence but the action is
    idempotent at the model-store level.
    """
    buffer = RetrainEvidenceBuffer(min_samples=1)
    action_executor = DummyActionExecutor()
    snapshot_store = SnapshotStore()

    decision_executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=action_executor,
        min_f1_improvement=0.05,
        dry_run=True,
        snapshot_store=snapshot_store,
    )

    decision = Decision(action="promote", reason="stability conditions satisfied")
    snapshot = DecisionSnapshot(
        decision_id=str(uuid.uuid4()),
        action=decision.action,
        timestamp=time.time(),
        status="pending",
    )

    await decision_executor.execute(decision=decision, snapshot=snapshot)

    assert snapshot.status == "executed"


@pytest.mark.asyncio
async def test_rollback_decision_executes_in_dry_run() -> None:
    """
    Rollback in dry_run must reach the executor path (snapshot written,
    action dispatched to executor) without touching the real model store.
    """
    buffer = RetrainEvidenceBuffer(min_samples=1)
    action_executor = DummyActionExecutor()

    decision_executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=action_executor,
        min_f1_improvement=0.05,
        dry_run=True,
    )

    decision = Decision(action="rollback", reason="catastrophic regression")
    snapshot = DecisionSnapshot(
        decision_id=str(uuid.uuid4()),
        action=decision.action,
        timestamp=time.time(),
        status="pending",
    )

    await decision_executor.execute(decision=decision, snapshot=snapshot)

    assert snapshot.status == "executed"
