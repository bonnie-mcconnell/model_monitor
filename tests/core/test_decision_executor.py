import asyncio
import pandas as pd
import pytest

from model_monitor.core.decision_executor import DecisionExecutor
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decisions import Decision
from model_monitor.core.model_actions import ModelAction
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer


class DummyActionExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[ModelAction, dict]] = []

    def execute(self, *, action: ModelAction, context: dict) -> None:
        self.calls.append((action, context))


@pytest.mark.asyncio
async def test_retrain_skipped_when_buffer_not_ready() -> None:
    buffer = RetrainEvidenceBuffer(min_samples=5)
    executor = DummyActionExecutor()

    decision_executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=executor,
        min_f1_improvement=0.01,
    )

    snapshot = DecisionSnapshot(
        decision_id="d1",
        action="retrain",
        timestamp=0.0,
        status="pending",
    )

    decision = Decision(action="retrain", reason="test")

    await decision_executor.execute(decision=decision, snapshot=snapshot)

    assert snapshot.status == "skipped"
    assert executor.calls == []


@pytest.mark.asyncio
async def test_retrain_dry_run_has_no_side_effects() -> None:
    buffer = RetrainEvidenceBuffer(min_samples=1)
    buffer._buffer.append(
        {
            "accuracy": 0.5,
            "f1": 0.4,
            "drift_score": 0.2,
            "trust_score": 0.3,
            "timestamp": 1.0,
        }
    )

    executor = DummyActionExecutor()

    decision_executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=executor,
        min_f1_improvement=0.01,
        dry_run=True,
    )

    snapshot = DecisionSnapshot(
        decision_id="d2",
        action="retrain",
        timestamp=1.0,
        status="pending",
    )

    decision = Decision(action="retrain", reason="test")

    await decision_executor.execute(decision=decision, snapshot=snapshot)

    assert snapshot.status == "executed"
    assert executor.calls == []


@pytest.mark.asyncio
async def test_promote_executes_once() -> None:
    buffer = RetrainEvidenceBuffer(min_samples=1)
    executor = DummyActionExecutor()

    decision_executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=executor,
        min_f1_improvement=0.01,
    )

    snapshot = DecisionSnapshot(
        decision_id="d3",
        action="promote",
        timestamp=2.0,
        status="pending",
    )

    decision = Decision(action="promote", reason="test")

    await decision_executor.execute(decision=decision, snapshot=snapshot)

    assert snapshot.status == "executed"
    assert executor.calls == [(ModelAction.PROMOTE, {})]
