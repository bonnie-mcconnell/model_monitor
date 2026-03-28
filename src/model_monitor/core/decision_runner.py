from __future__ import annotations

import time
from typing import Iterable, Optional, cast

from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.model_store import ModelStore


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
        model_store: ModelStore | None = None,
        now: float | None = None,
    ) -> list[Decision]:
        """
        Evaluate decisions for the specified aggregation windows.

        Args:
            windows: aggregation windows to evaluate (e.g. ["5m", "1h", "24h"])
            model_store: if provided, reads baseline_f1 from the active model's
                        promotion metadata. When None (e.g. in tests or before
                        first promotion), baseline falls back to current avg_f1,
                        which suppresses retrain/rollback but still fires on drift.
            now: override timestamp (primarily for testing)
        """
        now = now or time.time()
        decisions: list[Decision] = []
        recent_actions = self._load_recent_actions(limit=10)

        baseline_f1: float | None = None
        if model_store is not None:
            active_meta = model_store.get_active_metadata()
            baseline_f1 = active_meta.get("metrics", {}).get("baseline_f1")

        for window in windows:
            summary = self._summary_store.get(window)
            if summary is None:
                continue

            effective_baseline = baseline_f1 if baseline_f1 is not None else summary.avg_f1

            decision = self._engine.decide(
                batch_index=summary.n_batches,
                trust_score=summary.trust_score,   # now persisted by aggregation_loop
                f1=summary.avg_f1,
                f1_baseline=effective_baseline,
                drift_score=summary.avg_drift_score,
                recent_actions=recent_actions,
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