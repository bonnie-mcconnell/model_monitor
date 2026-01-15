# src/model_monitor/core/decision_executor.py
from __future__ import annotations

import asyncio
import logging
from typing import Any

from model_monitor.core.decisions import Decision
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.model_actions import ModelAction
from model_monitor.core.model_action_executor_protocol import (
    ModelActionExecutorProtocol,
)
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer

log = logging.getLogger(__name__)


class DecisionExecutor:
    """
    ASYNC orchestration layer.

    Responsibilities:
    - enforce concurrency rules
    - enforce retrain idempotency
    - manage execution state transitions
    - guard snapshot correctness

    Explicitly does NOT:
    - write metrics
    - persist snapshots
    """

    VALID_ACTIONS = {"none", "reject", "retrain", "promote", "rollback"}

    def __init__(
        self,
        *,
        retrain_buffer: RetrainEvidenceBuffer,
        action_executor: ModelActionExecutorProtocol,
        min_f1_improvement: float,
        dry_run: bool = False,
    ) -> None:
        self._retrain_buffer = retrain_buffer
        self._executor = action_executor
        self._min_f1_improvement = min_f1_improvement
        self._dry_run = dry_run
        self._retrain_lock = asyncio.Lock()

    async def execute(
        self,
        *,
        decision: Decision,
        snapshot: DecisionSnapshot,
        context: dict[str, Any] | None = None,
    ) -> None:
        if decision.action not in self.VALID_ACTIONS:
            raise ValueError(f"Unknown decision action: {decision.action}")

        action = ModelAction.from_decision(decision.action)

        # ----------------------------
        # No-op decisions
        # ----------------------------
        if action in {ModelAction.NONE, ModelAction.REJECT}:
            snapshot.status = "executed"
            return

        # ----------------------------
        # Retrain path
        # ----------------------------
        if action is ModelAction.RETRAIN:
            await self._handle_retrain(snapshot)
            return

        # ----------------------------
        # Promote / rollback
        # ----------------------------
        snapshot.status = "pending"

        if not self._dry_run:
            await asyncio.to_thread(
                self._executor.execute,
                action=action,
                context=context or {},
            )

        snapshot.status = "executed"

    async def _handle_retrain(self, snapshot: DecisionSnapshot) -> None:
        if not self._retrain_buffer.ready():
            snapshot.status = "skipped"
            return

        if self._retrain_lock.locked():
            snapshot.status = "skipped"
            return

        async with self._retrain_lock:
            df = self._retrain_buffer.consume()
            if df.empty:
                snapshot.status = "skipped"
                return

            retrain_key = self._retrain_buffer.retrain_key(df)

            if snapshot.retrain_key == retrain_key:
                snapshot.status = "skipped"
                return

            snapshot.retrain_key = retrain_key
            snapshot.status = "pending"

            try:
                if not self._dry_run:
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
