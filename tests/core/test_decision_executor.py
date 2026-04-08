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
        return None


@pytest.mark.asyncio
async def test_noop_decision_executes_without_side_effects() -> None:
    buffer = RetrainEvidenceBuffer(min_samples=1)
    executor = DummyActionExecutor()

    decision_executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=executor,
        min_f1_improvement=0.05,
    )

    decision = Decision(
        action="none",
        reason="no-op",
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

    decision = Decision(
        action="retrain",
        reason="low trust",
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

    assert snapshot.status == "skipped"
    assert executor.calls == []


@pytest.mark.asyncio
async def test_retrain_executes_when_buffer_ready() -> None:
    buffer = RetrainEvidenceBuffer(min_samples=1)
    buffer.add_summary(
        accuracy=0.5,
        f1=0.4,
        drift_score=0.6,
        trust_score=0.3,
        timestamp=time.time(),
    )

    executor = DummyActionExecutor()

    decision_executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=executor,
        min_f1_improvement=0.05,
        dry_run=True,
    )

    decision = Decision(
        action="retrain",
        reason="trust degraded",
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

    assert snapshot.status == "executed"
