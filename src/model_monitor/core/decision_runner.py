"""Synchronous control-plane: reads metrics, invokes engine, persists decisions.

Production path
---------------
The FastAPI server uses ``start_aggregation_loop`` (``monitoring/aggregation.py``),
which calls ``aggregate_once`` on a configurable poll interval.  That path handles
async execution, raw-data buffering, and executor wiring.

This module provides a **synchronous alternative** for:
- CLI scripts that need a one-shot decision pass without an async event loop
- Offline replay of stored summaries (e.g. debugging a past incident)
- Tests that drive the engine without spinning up the full aggregation stack

Both paths share the same ``DecisionEngine`` and produce identical decisions given
identical inputs.  ``DecisionRunner`` makes no model-lifecycle calls - it is read-only
with respect to models, and write-only with respect to the decision audit log.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import cast

from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.model_store import ModelStore


class DecisionRunner:
    """Synchronous, one-shot decision evaluator.

    Responsibilities:
    - Read the latest aggregated metric summaries from ``MetricsSummaryStore``
    - Invoke ``DecisionEngine`` with the correct inputs
    - Persist decisions to the audit log via ``DecisionStore``
    - Return decisions to the caller for inspection or logging

    Does NOT:
    - Execute model lifecycle actions (promote, rollback, retrain)
    - Manage execution state or snapshots
    - Spin up an async event loop
    """

    def __init__(
        self,
        *,
        decision_engine: DecisionEngine,
        summary_store: MetricsSummaryStore,
        decision_store: DecisionStore,
    ) -> None:
        self._engine = decision_engine
        self._summary_store = summary_store
        self._decision_store = decision_store

    def run_once(
        self,
        *,
        windows: Iterable[str],
        model_store: ModelStore | None = None,
        now: float | None = None,
    ) -> list[Decision]:
        """Evaluate decisions for the specified aggregation windows.

        Args:
            windows:     aggregation windows to evaluate (e.g. ``["5m", "1h"]``).
            model_store: when provided, reads ``baseline_f1`` from the active
                         model's promotion metadata so retrain/rollback decisions
                         compare against the real baseline.  When absent (cold
                         start or tests without a trained model), baseline falls
                         back to the current ``avg_f1``, which suppresses
                         retrain/rollback but still fires on drift.
            now:         override the current timestamp - used in tests to drive
                         time-windowed queries without sleeping.

        Returns:
            One ``Decision`` per window that had data.  Empty list when no
            summaries exist or all summaries are unpopulated (``n_batches == 0``).
        """
        now = now or time.time()
        decisions: list[Decision] = []
        recent_actions = self._load_recent_actions(limit=10)

        baseline_f1: float | None = None
        candidate_exists = False
        if model_store is not None:
            active_meta = model_store.get_active_metadata()
            baseline_f1 = active_meta.get("metrics", {}).get("baseline_f1")
            candidate_exists = model_store.has_candidate()

        for window in windows:
            summary = self._summary_store.get(window)
            if summary is None:
                continue

            # Guard against the cold-start race: MetricsSummaryORM rows are
            # created by the first aggregation pass, but their numeric columns
            # default to 0.0.  A trust_score of 0.0 would fire a spurious
            # retrain before any real data has been seen.
            if summary.n_batches == 0:
                continue

            effective_baseline = (
                baseline_f1 if baseline_f1 is not None else summary.avg_f1
            )

            decision = self._engine.decide(
                batch_index=summary.n_batches,
                trust_score=summary.trust_score,
                f1=summary.avg_f1,
                f1_baseline=effective_baseline,
                drift_score=summary.avg_drift_score,
                recent_actions=recent_actions,
                candidate_exists=candidate_exists,
            )

            self._decision_store.record(
                decision=decision,
                batch_index=summary.n_batches,
                trust_score=summary.trust_score,
                f1=summary.avg_f1,
                drift_score=summary.avg_drift_score,
            )

            decisions.append(decision)

        return decisions

    def _load_recent_actions(self, *, limit: int) -> list[DecisionType]:
        """Load recent decision actions for hysteresis and cooldown checks.

        Row ``action`` values are stored as plain strings; cast to ``DecisionType``
        at the persistence → domain boundary to keep the type system honest.
        """
        rows = self._decision_store.tail(limit=limit)
        return [cast(DecisionType, row.action) for row in rows]
