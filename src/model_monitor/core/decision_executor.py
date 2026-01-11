from __future__ import annotations

import asyncio
import logging

from model_monitor.core.decisions import Decision
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.model_actions import ModelAction
from model_monitor.core.model_action_executor import ModelActionExecutor
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer

log = logging.getLogger(__name__)


class DecisionExecutor:
    """
    ASYNC orchestration layer.

    Responsibilities:
    - enforce concurrency rules
    - enforce idempotency
    - manage execution state transitions
    - delegate side effects to ModelActionExecutor

    Owns execution correctness.
    """

    VALID_ACTIONS = {"none", "reject", "retrain", "promote", "rollback"}

    def __init__(
        self,
        *,
        retrain_buffer: RetrainEvidenceBuffer,
        action_executor: ModelActionExecutor,
        snapshot_store,
        min_f1_improvement: float,
    ) -> None:
        self._retrain_buffer = retrain_buffer
        self._executor = action_executor
        self._snapshot_store = snapshot_store
        self._min_f1_improvement = min_f1_improvement
        self._retrain_lock = asyncio.Lock()

    async def execute(
        self,
        *,
        decision: Decision,
        snapshot: DecisionSnapshot,
    ) -> None:
        if decision.action not in self.VALID_ACTIONS:
            raise ValueError(f"Unknown decision action: {decision.action}")

        action = ModelAction(decision.action)

        if action in {ModelAction.NONE, ModelAction.REJECT}:
            return

        if action is ModelAction.RETRAIN:
            await self._handle_retrain(snapshot)

        elif action is ModelAction.PROMOTE:
            await asyncio.to_thread(
                self._executor.execute,
                action=action,
                context={},
            )

        elif action is ModelAction.ROLLBACK:
            await asyncio.to_thread(
                self._executor.execute,
                action=action,
                context={},
            )

    async def _handle_retrain(self, snapshot: DecisionSnapshot) -> None:
        if not self._retrain_buffer.ready():
            log.info("Retrain skipped: insufficient evidence")
            return

        if self._retrain_lock.locked():
            log.info("Retrain already in progress; skipping")
            return

        async with self._retrain_lock:
            df = self._retrain_buffer.consume()
            if df.empty:
                return

            retrain_key = self._retrain_buffer.retrain_key(df)

            if self._snapshot_store.seen_retrain_key(retrain_key):
                log.info("Retrain skipped (idempotent): key=%s", retrain_key)
                return

            snapshot.retrain_key = retrain_key
            snapshot.status = "pending"
            self._snapshot_store.save(snapshot)

            try:
                await asyncio.to_thread(
                    self._executor.execute,
                    action=ModelAction.RETRAIN,
                    context={
                        "retrain_df": df,
                        "min_f1_improvement": self._min_f1_improvement,
                    },
                )
                snapshot.status = "executed"

            except Exception:
                snapshot.status = "failed"
                raise

            finally:
                self._snapshot_store.save(snapshot)
