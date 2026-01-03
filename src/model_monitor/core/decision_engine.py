from __future__ import annotations

from model_monitor.core.decisions import Decision
from model_monitor.config.settings import AppConfig


class DecisionEngine:
    """
    Converts monitoring signals into operational decisions.

    This class contains POLICY ONLY.
    It does NOT execute actions, perform I/O, or touch storage.
    """

    def __init__(self, config: AppConfig):
        self.cfg = config
        self.last_retrain_batch_index: int | None = None

    def decide(
        self,
        *,
        batch_index: int,
        f1: float,
        f1_baseline: float,
        drift_score: float,
        recent_actions: list[str] | None = None,
    ) -> Decision:
        """
        Decide what operational action (if any) should be taken
        based on model performance and data drift signals.
        """

        f1_drop = f1_baseline - f1

        # --------------------------------------------------
        # Hard drift guardrail → reject
        # --------------------------------------------------
        if drift_score >= self.cfg.drift.psi_threshold:
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
            return Decision(
                action="rollback",
                reason="Catastrophic performance regression",
                metadata={
                    "f1_drop": f1_drop,
                    "baseline_f1": f1_baseline,
                    "current_f1": f1,
                    "rollback_to": "previous",
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
            return Decision(
                action="rollback",
                reason="Severe regression without corresponding drift",
                metadata={
                    "f1_drop": f1_drop,
                    "baseline_f1": f1_baseline,
                    "current_f1": f1,
                    "drift_score": drift_score,
                    "rollback_to": "previous",
                },
            )

        # --------------------------------------------------
        # Sustained degradation → retrain
        # --------------------------------------------------
        if f1_drop >= self.cfg.retrain.min_f1_gain:
            if (
                self.last_retrain_batch_index is not None
                and batch_index - self.last_retrain_batch_index
                < self.cfg.retrain.cooldown_batches
            ):
                return Decision(
                    action="none",
                    reason="Retrain cooldown active",
                    metadata={
                        "cooldown_batches": self.cfg.retrain.cooldown_batches,
                    },
                )

            self.last_retrain_batch_index = batch_index
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
        # Promotion guardrail (N stable batches)
        # --------------------------------------------------
        if recent_actions is not None:
            n = self.cfg.retrain.min_stable_batches
            if (
                len(recent_actions) >= n
                and all(a == "none" for a in recent_actions[-n:])
            ):
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
        return Decision(
            action="none",
            reason="System operating within thresholds",
            metadata={},
        )
