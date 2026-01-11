from __future__ import annotations

from typing import Sequence

from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.config.settings import AppConfig


class DecisionEngine:
    """
    Converts monitoring signals into operational decisions.

    POLICY ONLY:
    - No I/O
    - No persistence
    - No model mutation

    Maintains minimal ephemeral state for retrain cooldown tracking.
    Intended to be scoped per execution context.
    """

    def __init__(self, config: AppConfig):
        self.cfg = config
        self._last_retrain_batch: int | None = None

    def decide(
        self,
        *,
        batch_index: int,
        trust_score: float,
        f1: float,
        f1_baseline: float,
        drift_score: float,
        recent_actions: Sequence[DecisionType] | None = None,
    ) -> Decision:
        # -----------------------------
        # Guardrails
        # -----------------------------
        assert 0.0 <= trust_score <= 1.0, "trust_score must be in [0,1]"
        assert f1 >= 0.0, "f1 must be non-negative"
        assert f1_baseline >= 0.0, "baseline f1 must be non-negative"

        f1_drop = f1_baseline - f1

        # -----------------------------
        # 1. Severe drift → reject
        # -----------------------------
        if drift_score >= self.cfg.drift.psi_threshold:
            return Decision(
                action="reject",
                reason="Severe feature drift detected",
                metadata={
                    "drift_score": drift_score,
                    "threshold": self.cfg.drift.psi_threshold,
                    "trust_score": trust_score,
                },
            )

        # -----------------------------
        # 2. Catastrophic regression → rollback
        # -----------------------------
        if f1_drop >= self.cfg.rollback.max_f1_drop:
            return Decision(
                action="rollback",
                reason="Catastrophic performance regression detected",
                metadata={
                    "f1_drop": f1_drop,
                    "baseline_f1": f1_baseline,
                    "current_f1": f1,
                    "trust_score": trust_score,
                },
            )

        # -----------------------------
        # 3. Sustained degradation → retrain
        # -----------------------------
        if f1_drop >= self.cfg.retrain.min_f1_gain:
            if self._last_retrain_batch is not None:
                since_last = batch_index - self._last_retrain_batch
                if since_last < self.cfg.retrain.cooldown_batches:
                    return Decision(
                        action="none",
                        reason="Retrain cooldown active",
                        metadata={
                            "cooldown_batches": self.cfg.retrain.cooldown_batches,
                            "batches_since_last_retrain": since_last,
                        },
                    )

            self._last_retrain_batch = batch_index
            return Decision(
                action="retrain",
                reason="Sustained performance degradation detected",
                metadata={
                    "f1_drop": f1_drop,
                    "baseline_f1": f1_baseline,
                    "current_f1": f1,
                    "trust_score": trust_score,
                },
            )

        # -----------------------------
        # 4. Stability → promote
        # -----------------------------
        if recent_actions is not None:
            n = self.cfg.retrain.min_stable_batches
            if len(recent_actions) >= n and all(a == "none" for a in recent_actions[-n:]):
                return Decision(
                    action="promote",
                    reason="Promotion stability conditions satisfied",
                    metadata={"stable_batches": n},
                )

        # -----------------------------
        # 5. Default → none
        # -----------------------------
        return Decision(
            action="none",
            reason="System operating within thresholds",
            metadata={"trust_score": trust_score},
        )
