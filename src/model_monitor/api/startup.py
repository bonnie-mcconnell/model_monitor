from __future__ import annotations

import asyncio
import logging
from typing import Optional

from model_monitor.config.settings import load_config
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decision_runner import DecisionRunner
from model_monitor.monitoring.aggregation import start_aggregation_loop
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)
from model_monitor.storage.model_store import ModelStore

log = logging.getLogger(__name__)


def start_background_loops(
    *,
    poll_interval: int = 60,
    decision_interval: int = 60,
) -> None:
    """
    Start background aggregation and decision loops.

    Intended to be called during FastAPI startup.
    """

    # ---- shared stores ----
    metrics_store = MetricsStore()
    summary_store = MetricsSummaryStore()
    history_store = MetricsSummaryHistoryStore()
    decision_store = DecisionStore()
    model_store = ModelStore()

    retrain_buffer = RetrainEvidenceBuffer(min_samples=5)

    decision_engine = DecisionEngine(load_config())
    decision_runner = DecisionRunner(
        decision_engine=decision_engine,
        summary_store=summary_store,
        decision_store=decision_store,
    )

    async def aggregation_loop() -> None:
        log.info(
            "Starting metrics aggregation loop (interval=%s seconds)",
            poll_interval,
        )
        await start_aggregation_loop(
            metrics_store=metrics_store,
            summary_store=summary_store,
            history_store=history_store,
            retrain_buffer=retrain_buffer,
            model_store=model_store,
            poll_interval=poll_interval,
        )

    async def decision_loop() -> None:
        log.info(
            "Starting decision evaluation loop (interval=%s seconds)",
            decision_interval,
        )
        while True:
            try:
                decision_runner.run_once(
                    windows=["5m", "1h", "24h"]
                )
            except Exception:
                log.exception("Decision loop failed")
            await asyncio.sleep(decision_interval)

    async def runner() -> None:
        await asyncio.gather(
            aggregation_loop(),
            decision_loop(),
        )

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(runner())
    except RuntimeError:
        asyncio.run(runner())
