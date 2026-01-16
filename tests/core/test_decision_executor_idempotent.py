import time
import uuid
import pytest

from model_monitor.core.decision_executor import DecisionExecutor
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decisions import Decision
from model_monitor.core.model_actions import ModelAction
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer


class RecordingActionExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, *, action: ModelAction, context):
        self.calls.append((action, context))


@pytest.mark.asyncio
async def test_decision_executor_is_idempotent():
    """
    Executing the same decision snapshot twice must not cause
    duplicate side effects.
    """

    buffer = RetrainEvidenceBuffer(min_samples=1)
    buffer.add_summary(
        accuracy=0.5,
        f1=0.4,
        drift_score=0.6,
        trust_score=0.3,
        timestamp=time.time(),
    )

    action_executor = RecordingActionExecutor()

    executor = DecisionExecutor(
        retrain_buffer=buffer,
        action_executor=action_executor,
        min_f1_improvement=0.05,
        dry_run=False,
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

    # First execution
    await executor.execute(decision=decision, snapshot=snapshot)

    # Second execution (retry)
    await executor.execute(decision=decision, snapshot=snapshot)

    # Action must only be executed once
    assert len(action_executor.calls) == 1

    # Snapshot should be stable
    assert snapshot.status == "executed"
