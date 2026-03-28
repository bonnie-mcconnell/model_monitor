from __future__ import annotations

import asyncio
import logging

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
) -> None:
    """
    Start background aggregation and decision loops.
    Intended to be called during FastAPI startup.
    """
    metrics_store = MetricsStore()
    summary_store = MetricsSummaryStore()
    history_store = MetricsSummaryHistoryStore()
    decision_store = DecisionStore()
    model_store = ModelStore()
    retrain_buffer = RetrainEvidenceBuffer(min_samples=5)

    async def aggregation_loop() -> None:
        log.info("Starting aggregation loop (interval=%ss)", poll_interval)
        await start_aggregation_loop(
            metrics_store=metrics_store,
            summary_store=summary_store,
            history_store=history_store,
            retrain_buffer=retrain_buffer,
            model_store=model_store,
            decision_store=decision_store,
            poll_interval=poll_interval,
        )

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(aggregation_loop())
    except RuntimeError:
        asyncio.run(aggregation_loop())