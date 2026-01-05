from __future__ import annotations

from typing import Sequence

from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.config.settings import AppConfig


class DecisionEngine:
    """
    Converts monitoring signals into operational decisions.

    POLICY ONLY:
    - no I/O
    - no persistence
    - no side effects except internal cooldown tracking
    """

    def __init__(self, config: AppConfig):
        self.cfg = config
        self._last_retrain_batch: int | None = None
        self._last_action: DecisionType | None = None

    def decide(
        self,
        *,
        batch_index: int,
        f1: float,
        f1_baseline: float,
        drift_score: float,
        recent_actions: Sequence[DecisionType] | None = None,
    ) -> Decision:
        """
        Decide what operational action (if any) should be taken.
        """

        f1_drop = f1_baseline - f1

        # --------------------------------------------------
        # Hard drift guardrail → reject traffic
        # --------------------------------------------------
        if drift_score >= self.cfg.drift.psi_threshold:
            self._last_action = "reject"
            return Decision(
                action="reject",
                reason="Severe feature drift detected",
                metadata={
                    "drift_score": drift_score,
                    "threshold": self.cfg.drift.psi_threshold,
                },
            )

        # --------------------------------------------------
        # Catastrophic regression → rollback
        # --------------------------------------------------
        if f1_drop >= self.cfg.rollback.max_f1_drop:
            self._last_action = "rollback"
            return Decision(
                action="rollback",
                reason="Catastrophic performance regression",
                metadata={
                    "f1_drop": f1_drop,
                    "baseline_f1": f1_baseline,
                    "current_f1": f1,
                    "target_version": "previous",
                },
            )

        # --------------------------------------------------
        # Suspected bad rollout (regression without drift)
        # --------------------------------------------------
        severe_regression = self.cfg.retrain.min_f1_gain * 2

        if (
            f1_drop >= severe_regression
            and drift_score < self.cfg.drift.psi_threshold * 0.5
        ):
            self._last_action = "rollback"
            return Decision(
                action="rollback",
                reason="Severe regression without corresponding drift",
                metadata={
                    "f1_drop": f1_drop,
                    "baseline_f1": f1_baseline,
                    "current_f1": f1,
                    "drift_score": drift_score,
                    "target_version": "previous",
                },
            )

        # --------------------------------------------------
        # Sustained degradation → retrain
        # --------------------------------------------------
        if f1_drop >= self.cfg.retrain.min_f1_gain:
            if self._last_retrain_batch is not None:
                since_last = batch_index - self._last_retrain_batch
                if since_last < self.cfg.retrain.cooldown_batches:
                    self._last_action = "none"
                    return Decision(
                        action="none",
                        reason="Retrain cooldown active",
                        metadata={
                            "cooldown_batches": self.cfg.retrain.cooldown_batches,
                            "batches_since_last_retrain": since_last,
                        },
                    )

            self._last_retrain_batch = batch_index
            self._last_action = "retrain"
            return Decision(
                action="retrain",
                reason="Sustained performance degradation",
                metadata={
                    "f1_drop": f1_drop,
                    "baseline_f1": f1_baseline,
                    "current_f1": f1,
                },
            )

        # --------------------------------------------------
        # Promotion guardrail (N stable batches, no recent ops)
        # --------------------------------------------------
        if recent_actions is not None:
            n = self.cfg.retrain.min_stable_batches
            if (
                len(recent_actions) >= n
                and all(a == "none" for a in recent_actions[-n:])
                and self._last_action == "none"
            ):
                self._last_action = "promote"
                return Decision(
                    action="promote",
                    reason="Promotion stability conditions satisfied",
                    metadata={
                        "stable_batches": n,
                    },
                )

        # --------------------------------------------------
        # Stable system
        # --------------------------------------------------
        self._last_action = "none"
        return Decision(
            action="none",
            reason="System operating within thresholds",
            metadata={},
        )
