from __future__ import annotations

import time
from typing import Iterable, Optional, cast

from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.decision_store import DecisionStore


class DecisionRunner:
    """
    Control-plane decision orchestrator.

    Responsibilities:
    - Read latest aggregated metric summaries
    - Invoke DecisionEngine
    - Persist decisions to the audit log

    Explicitly does NOT:
    - Execute model actions
    - Manage execution state
    - Create DecisionSnapshots
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
        now: Optional[float] = None,
    ) -> list[Decision]:
        """
        Evaluate decisions for the specified aggregation windows.

        Args:
            windows: aggregation windows to evaluate (e.g. ["5m", "1h", "24h"])
            now: override timestamp (primarily for testing)

        Returns:
            List of decisions produced.
        """
        now = now or time.time()
        decisions: list[Decision] = []

        recent_actions = self._load_recent_actions(limit=10)

        for window in windows:
            summary = self._summary_store.get(window)
            if summary is None:
                continue

            decision = self._engine.decide(
                batch_index=summary.n_batches,
                trust_score=summary.avg_accuracy,  # trust_score not persisted here
                f1=summary.avg_f1,
                f1_baseline=summary.avg_f1,  # baseline wiring later
                drift_score=summary.avg_drift_score,
                recent_actions=recent_actions,
            )

            self._decision_store.record(
                decision=decision,
                batch_index=summary.n_batches,
                trust_score=None,  # not stored yet
                f1=summary.avg_f1,
                drift_score=summary.avg_drift_score,
            )

            decisions.append(decision)

        return decisions

    def _load_recent_actions(self, *, limit: int) -> list[DecisionType]:
        """
        Load recent decision actions for hysteresis / cooldown logic.

        DB values are stored as strings; we explicitly cast to DecisionType
        at the persistence → domain boundary.
        """
        rows = self._decision_store.tail(limit=limit)

        actions: list[DecisionType] = []
        for row in rows:
            # row.action is a str, but constrained by invariant to DecisionType
            actions.append(cast(DecisionType, row.action))

        return actions
