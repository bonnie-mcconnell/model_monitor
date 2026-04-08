"""Async execution layer with retrain locking and crash-safe idempotency."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.core.decisions import Decision
from model_monitor.core.model_action_executor_protocol import (
    ModelActionExecutorProtocol,
)
from model_monitor.core.model_actions import ModelAction
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer

log = logging.getLogger(__name__)


class DecisionExecutor:
    """
    Async orchestration layer.

    Responsibilities:
    - Enforce concurrency rules (asyncio.Lock for retrain)
    - Enforce retrain idempotency (SHA-256 retrain_key)
    - Persist snapshots before execution (write-ahead log)
    - Manage execution state transitions

    Does NOT:
    - Write metrics
    - Make policy decisions

    Crash safety:
        Before executing a retrain, the snapshot is written to SnapshotStore.
        On restart, pending snapshots are detected and their retrain_keys
        used to suppress duplicate execution. This closes the gap where a
        crash between key assignment and execution could trigger a duplicate
        retrain.
    """

    VALID_ACTIONS = {"none", "reject", "retrain", "promote", "rollback"}

    def __init__(
        self,
        *,
        retrain_buffer: RetrainEvidenceBuffer,
        action_executor: ModelActionExecutorProtocol,
        min_f1_improvement: float,
        dry_run: bool = False,
        snapshot_store: Any | None = None,
    ) -> None:
        self._retrain_buffer = retrain_buffer
        self._executor = action_executor
        self._min_f1_improvement = min_f1_improvement
        self._dry_run = dry_run
        self._snapshot_store = snapshot_store  # None = crash recovery disabled
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

        if action in {ModelAction.NONE, ModelAction.REJECT}:
            snapshot.status = "executed"
            return

        if action is ModelAction.RETRAIN:
            await self._handle_retrain(snapshot)
            return

        # Promote / rollback - write snapshot before touching the model store
        if self._snapshot_store is not None:
            self._snapshot_store.write(snapshot)

        snapshot.status = "pending"
        if not self._dry_run:
            await asyncio.to_thread(
                self._executor.execute,
                action=action,
                context=context or {},
            )
        snapshot.status = "executed"

        if self._snapshot_store is not None:
            self._snapshot_store.update_status(snapshot)

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

            # Check in-memory idempotency (same process)
            if snapshot.retrain_key == retrain_key:
                snapshot.status = "skipped"
                return

            # Check durable idempotency (cross-restart)
            if (
                self._snapshot_store is not None
                and self._snapshot_store.is_retrain_key_known(retrain_key)
            ):
                log.warning(
                    "retrain_skipped_duplicate_key",
                    extra={"retrain_key": retrain_key[:16]},
                )
                snapshot.status = "skipped"
                return

            snapshot.retrain_key = retrain_key
            snapshot.status = "pending"

            # Write-ahead: persist BEFORE execution.
            # If the process crashes here, the key is in the DB.
            # On restart, is_retrain_key_known() returns True and skips.
            if self._snapshot_store is not None:
                self._snapshot_store.write(snapshot)

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
            finally:
                if self._snapshot_store is not None:
                    self._snapshot_store.update_status(snapshot)
