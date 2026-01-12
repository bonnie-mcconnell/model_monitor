import asyncio
import logging
from typing import Optional

from model_monitor.monitoring.aggregation import start_aggregation_loop
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)
from model_monitor.storage.model_store import ModelStore

log = logging.getLogger(__name__)


def start_background_aggregation_loop(
    *,
    metrics_store: Optional[MetricsStore] = None,
    summary_store: Optional[MetricsSummaryStore] = None,
    history_store: Optional[MetricsSummaryHistoryStore] = None,
    model_store: Optional[ModelStore] = None,
    poll_interval: int = 60,
) -> None:
    """
    Start the background metrics aggregation loop.

    Intended to be called during FastAPI startup.
    """
    metrics_store = metrics_store or MetricsStore()
    summary_store = summary_store or MetricsSummaryStore()
    history_store = history_store or MetricsSummaryHistoryStore()
    model_store = model_store or ModelStore()
    retrain_buffer = RetrainEvidenceBuffer(min_samples=5)

    async def _runner() -> None:
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

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_runner())
    except RuntimeError:
        asyncio.run(_runner())
