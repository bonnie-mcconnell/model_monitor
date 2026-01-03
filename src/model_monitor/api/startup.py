import asyncio
import logging

from model_monitor.monitoring.background_aggregator import MetricsAggregator
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore


log = logging.getLogger(__name__)


async def start_aggregation_loop() -> None:
    metrics_store = MetricsStore()
    summary_store = MetricsSummaryStore()
    aggregator = MetricsAggregator(metrics_store, summary_store)

    while True:
        try:
            aggregator.run_once()
        except Exception:
            log.exception("Metrics aggregation failed")

        await asyncio.sleep(60)  # run every minute
