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
    - no side effects except cooldown tracking
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
        f1_drop = f1_baseline - f1

        # --------------------------------------------------
        # 1. Hard drift guardrail → reject
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
        # 2. Catastrophic regression → rollback
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
                },
            )

        # --------------------------------------------------
        # 3. Sustained degradation → retrain
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
        # 4. Promotion guardrail
        # --------------------------------------------------
        if recent_actions is not None:
            n = self.cfg.retrain.min_stable_batches
            if (
                len(recent_actions) >= n
                and all(a == "none" for a in recent_actions[-n:])
            ):
                self._last_action = "promote"
                return Decision(
                    action="promote",
                    reason="Promotion stability conditions satisfied",
                    metadata={"stable_batches": n},
                )

        # --------------------------------------------------
        # 5. Stable system
        # --------------------------------------------------
        self._last_action = "none"
        return Decision(
            action="none",
            reason="System operating within thresholds",
            metadata={},
        )
