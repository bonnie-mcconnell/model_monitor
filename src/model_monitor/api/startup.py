"""Background loop initialization for the FastAPI lifespan."""

from __future__ import annotations

import asyncio
import logging

from model_monitor.config.settings import load_config
from model_monitor.monitoring.aggregation import start_aggregation_loop
from model_monitor.monitoring.raw_data_buffer import RawDataBuffer
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.storage.alert_store import AlertStore
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.model_store import ModelStore

log = logging.getLogger(__name__)


def start_background_loops(
    *,
    poll_interval: int = 60,
) -> None:
    """Start background aggregation and decision loops.

    Intended to be called during FastAPI startup.

    Reads ``min_samples`` from the application config (``retrain.yaml``) so
    both the evidence buffer and the raw data buffer respect the same threshold
    as the decision engine - not a hardcoded constant that bypasses config.

    Both buffers are constructed here and passed into the aggregation loop:
    - ``RetrainEvidenceBuffer``: accumulates aggregated monitoring signals that
      determine *when* to retrain.  Gated on ``evidence_window`` (default 3),
      not ``min_samples`` - the two thresholds govern different concerns and
      must not be conflated.  ``min_samples`` governs the raw training data
      quantity check in ``DefaultModelActionExecutor``.
    - ``RawDataBuffer``: accumulates labeled (X, y) pairs that provide *what
      to train on* when a retrain fires.
    """
    cfg = load_config()

    metrics_store = MetricsStore()
    summary_store = MetricsSummaryStore()
    history_store = MetricsSummaryHistoryStore()
    decision_store = DecisionStore()
    model_store = ModelStore()
    alert_store = AlertStore()
    retrain_buffer = RetrainEvidenceBuffer(min_samples=cfg.retrain.evidence_window)
    raw_data_buffer = RawDataBuffer(max_rows=50_000)

    async def aggregation_loop() -> None:
        log.info("Starting aggregation loop (interval=%ss)", poll_interval)
        await start_aggregation_loop(
            metrics_store=metrics_store,
            summary_store=summary_store,
            history_store=history_store,
            retrain_buffer=retrain_buffer,
            model_store=model_store,
            decision_store=decision_store,
            alert_store=alert_store,
            raw_data_buffer=raw_data_buffer,
            poll_interval=poll_interval,
        )

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(aggregation_loop())
    except RuntimeError:
        asyncio.run(aggregation_loop())
