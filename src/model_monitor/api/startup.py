import asyncio
import logging
from typing import Optional

from model_monitor.monitoring.aggregation import start_aggregation_loop
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore

log = logging.getLogger(__name__)


async def _run_aggregation_loop(
    metrics_store: Optional[MetricsStore] = None,
    summary_store: Optional[MetricsSummaryStore] = None,
    poll_interval: int = 60,
) -> None:
    """
    Async loop to continuously aggregate metrics in the background.

    Args:
        metrics_store: Optional pre-initialized MetricsStore instance.
        summary_store: Optional pre-initialized MetricsSummaryStore instance.
        poll_interval: Time in seconds between aggregation runs.
    """
    metrics_store = metrics_store or MetricsStore()
    summary_store = summary_store or MetricsSummaryStore()

    log.info("Starting metrics aggregation loop with interval %s seconds", poll_interval)

    while True:
        try:
            await start_aggregation_loop(poll_interval=poll_interval)
        except Exception:
            log.exception("Metrics aggregation iteration failed")
        await asyncio.sleep(poll_interval)


def start_background_aggregation_loop(
    metrics_store: Optional[MetricsStore] = None,
    summary_store: Optional[MetricsSummaryStore] = None,
    poll_interval: int = 60,
) -> None:
    """
    Entry point to start the aggregation loop in a separate asyncio task.

    This function is intended to be called during FastAPI startup.
    """
    loop = asyncio.get_event_loop()

    try:
        loop.create_task(
            _run_aggregation_loop(
                metrics_store=metrics_store,
                summary_store=summary_store,
                poll_interval=poll_interval,
            )
        )
        log.info("Metrics aggregation task scheduled successfully")
    except RuntimeError:
        log.warning("No running event loop; starting a new one")
        asyncio.run(
            _run_aggregation_loop(
                metrics_store=metrics_store,
                summary_store=summary_store,
                poll_interval=poll_interval,
            )
        )
