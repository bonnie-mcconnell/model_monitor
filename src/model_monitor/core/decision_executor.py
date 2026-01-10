from __future__ import annotations

import asyncio
import logging

from model_monitor.core.decisions import Decision
from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.training.retrain_pipeline import RetrainPipeline
from model_monitor.storage import model_store

log = logging.getLogger(__name__)


class DecisionExecutor:
    """
    Executes side-effectful consequences of decisions.

    Responsibilities:
    - retraining
    - promotion
    - rollback

    ASYNC ONLY.
    Never blocks aggregation or decision logic.
    """

    def __init__(
        self,
        *,
        retrain_buffer: RetrainEvidenceBuffer,
        retrain_pipeline: RetrainPipeline,
    ) -> None:
        self._retrain_buffer = retrain_buffer
        self._retrain_pipeline = retrain_pipeline

        # Prevent overlapping retrain jobs
        self._retrain_lock = asyncio.Lock()

    async def execute(
        self,
        *,
        decision: Decision,
        snapshot: DecisionSnapshot,
    ) -> None:
        action = decision.action

        if action == "retrain":
            await self._handle_retrain(snapshot)

        elif action == "promote":
            await self._handle_promote()

        elif action == "rollback":
            await self._handle_rollback()

        # "reject" and "none" have no side effects

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

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

            log.info("Starting retrain job (samples=%d)", len(df))

            try:
                current_model = model_store.load_current()
            except FileNotFoundError:
                current_model = None
                log.info("No active model found; training baseline model")

            result = await asyncio.to_thread(
                self._retrain_pipeline.run,
                retrain_df=df,
                current_model=current_model,
                min_f1_improvement=0.01,
            )

            if result.promotion.promoted and result.candidate_model is not None:
                model_store.save_candidate(result.candidate_model)
                model_store.promote_candidate(
                    metrics={
                        "f1": result.promotion.candidate_f1,
                        "improvement": result.promotion.improvement,
                        "n_samples": result.n_samples,
                    }
                )
                log.info(
                    "Retrain successful, model promoted (ΔF1=%.4f)",
                    result.promotion.improvement,
                )
            else:
                log.info(
                    "Retrain completed, candidate rejected (reason=%s)",
                    result.promotion.reason,
                )

    async def _handle_promote(self) -> None:
        log.info("Promoting existing candidate model")
        model_store.promote_candidate()

    async def _handle_rollback(self) -> None:
        active_version = model_store.get_active_version()
        archived = model_store.list_versions()

        if not archived:
            raise RuntimeError("No archived models available for rollback")

        # Roll back to most recent archived version that is not active
        target = next(
            (v["version"] for v in archived if v["version"] != active_version),
            None,
        )

        if target is None:
            raise RuntimeError("No valid rollback target found")

        model_store.rollback(version=target)
        log.warning("Rolled back model to version %s", target)
