"""Pure policy engine: converts monitoring signals into operational decisions.

The engine is a pure function of its inputs plus minimal ephemeral state.
No I/O, no persistence, no model mutation - all side-effects live in the
executor layer.  This means the engine is:
  - Fully unit-testable without mocks
  - Replayable: same inputs always produce the same output
  - Auditable: every decision is explained via ``Decision.reason``

Retrain circuit breaker
-----------------------
The cooldown prevents rapid cycling within a single run.  But what if every
retrain produces a model that *still* drifts?  Without a circuit breaker the
engine retries indefinitely, burning compute and masking an underlying data
quality problem.

``max_retrain_attempts`` caps the total number of retrain decisions per engine
lifetime.  When the counter reaches the cap the engine emits ``system_error``
instead of ``retrain`` and halts.  An operator resets the counter by calling
``reset_retrain_counter()`` after diagnosing the root cause.

Set ``max_retrain_attempts = 0`` in config to disable the circuit breaker.

Behavioral signal integration
------------------------------
Behavioral signals enter via the trust_score argument.  The trust score already
incorporates a behavioral_violation_rate component (see monitoring/trust_score.py),
so the engine responds to behavioral degradation through the same thresholds that
govern performance degradation - no separate code path required.
"""

from __future__ import annotations

from collections.abc import Sequence

from model_monitor.config.settings import AppConfig
from model_monitor.core.decisions import Decision, DecisionType


class DecisionEngine:
    """Convert monitoring signals into operational decisions.

    POLICY ONLY - no I/O, no persistence, no model mutation.

    Maintains two pieces of ephemeral state:
      _last_retrain_batch:  batch index of the most recent retrain decision.
                            Used for within-process cooldown enforcement.
      _retrain_attempt_count: cumulative retrain decisions fired this lifetime.
                            Feeds the circuit breaker.

    Both counters can be reset via ``reset_retrain_counter()``, which an
    operator should call after diagnosing a persistent drift/retrain loop.
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self._last_retrain_batch: int | None = None
        self._retrain_attempt_count: int = 0

    def reset_retrain_counter(self) -> None:
        """Reset both the cooldown tracker and the circuit-breaker counter.

        Call this after diagnosing and fixing the root cause of a runaway
        retrain loop, to allow the engine to issue retrain decisions again.
        """
        self._last_retrain_batch = None
        self._retrain_attempt_count = 0

    @property
    def retrain_attempt_count(self) -> int:
        """Total number of retrain decisions issued since construction or last reset."""
        return self._retrain_attempt_count

    def _fire_retrain(self, batch_index: int, reason: str, metadata: dict) -> Decision:
        """Record a retrain attempt, check circuit breaker, return Decision.

        This is the single choke-point through which every retrain decision
        passes.  Centralising the counter increment here ensures the circuit
        breaker cannot be bypassed by adding new retrain code paths.

        Returns:
            Decision with action=RETRAIN if under the cap, else SYSTEM_ERROR.
        """
        max_attempts = self.cfg.retrain.max_retrain_attempts
        if max_attempts > 0 and self._retrain_attempt_count >= max_attempts:
            return Decision(
                action=DecisionType.SYSTEM_ERROR,
                reason=(
                    f"Retrain circuit breaker open: {self._retrain_attempt_count} attempts "
                    f"reached max_retrain_attempts={max_attempts}. "
                    "Call engine.reset_retrain_counter() after fixing the root cause."
                ),
                metadata={
                    **metadata,
                    "retrain_attempt_count": self._retrain_attempt_count,
                    "max_retrain_attempts": max_attempts,
                },
            )

        self._retrain_attempt_count += 1
        self._last_retrain_batch = batch_index
        return Decision(
            action=DecisionType.RETRAIN,
            reason=reason,
            metadata={**metadata, "retrain_attempt_count": self._retrain_attempt_count},
        )

    def decide(
        self,
        *,
        batch_index: int,
        trust_score: float,
        f1: float,
        f1_baseline: float,
        drift_score: float,
        recent_actions: Sequence[DecisionType] | None = None,
        candidate_exists: bool = False,
    ) -> Decision:
        """Produce a single operational decision for this monitoring batch.

        Args:
            batch_index:     Monotonically increasing batch counter.  Used for
                             cooldown arithmetic.
            trust_score:     Weighted composite health score in [0, 1].
            f1:              Current batch F1 score.
            f1_baseline:     F1 score at training / promotion time.  Fixed reference.
            drift_score:     Current PSI drift score.
            recent_actions:  List of recent DecisionType values from the action
                             history store.  Used for durable cooldown and
                             reject-escalation checks.
            candidate_exists: True when a staged candidate model is available
                             for promotion evaluation.

        Returns:
            Decision with action, reason, and metadata.

        Raises:
            ValueError: if trust_score is outside [0, 1] or f1 values are negative.
        """
        if not (0.0 <= trust_score <= 1.0):
            raise ValueError(f"trust_score must be in [0,1], got {trust_score}")
        if f1 < 0.0:
            raise ValueError(f"f1 must be non-negative, got {f1}")
        if f1_baseline < 0.0:
            raise ValueError(f"f1_baseline must be non-negative, got {f1_baseline}")

        recent_actions = list(recent_actions or [])
        f1_drop = f1_baseline - f1

        # ------------------------------------------------------------------
        # 1. Severe drift → reject (escalates to retrain if stuck)
        # ------------------------------------------------------------------
        # Perpetual rejection is an operational dead-end: the model never
        # recovers without new training data.  When the last N consecutive
        # actions were all "reject" (N = cooldown_batches) we escalate to
        # retrain so the pipeline can build a model on the new distribution.
        if drift_score >= self.cfg.drift.psi_threshold:
            n = self.cfg.retrain.cooldown_batches
            reject_window = recent_actions[-n:]
            cooldown_clear = (
                self._last_retrain_batch is None
                or batch_index - self._last_retrain_batch >= n
            )
            if (
                len(reject_window) >= n
                and all(a == DecisionType.REJECT for a in reject_window)
                and cooldown_clear
            ):
                return self._fire_retrain(
                    batch_index,
                    reason="Sustained severe drift: escalating reject → retrain to recover",
                    metadata={
                        "drift_score": drift_score,
                        "threshold": self.cfg.drift.psi_threshold,
                        "consecutive_rejects": len(reject_window),
                        "trust_score": trust_score,
                    },
                )
            return Decision(
                action=DecisionType.REJECT,
                reason="Severe feature drift detected",
                metadata={
                    "drift_score": drift_score,
                    "threshold": self.cfg.drift.psi_threshold,
                    "trust_score": trust_score,
                },
            )

        # ------------------------------------------------------------------
        # 2. Catastrophic regression → rollback
        # ------------------------------------------------------------------
        # Require at least min_stable_batches observations before allowing
        # rollback.  Single-batch F1 is noisy on small/imbalanced batches;
        # triggering rollback before the model has been observed over multiple
        # batches produces false positives on cold start.
        if (
            f1_drop >= self.cfg.rollback.max_f1_drop
            and batch_index >= self.cfg.retrain.min_stable_batches
        ):
            return Decision(
                action=DecisionType.ROLLBACK,
                reason="Catastrophic performance regression detected",
                metadata={
                    "f1_drop": f1_drop,
                    "baseline_f1": f1_baseline,
                    "current_f1": f1,
                    "trust_score": trust_score,
                },
            )

        # ------------------------------------------------------------------
        # 3. Sustained degradation → retrain (with hysteresis + circuit breaker)
        # ------------------------------------------------------------------
        if f1_drop >= self.cfg.retrain.min_f1_gain:
            # Ephemeral cooldown: within-process, does not survive restarts.
            if self._last_retrain_batch is not None:
                since_last = batch_index - self._last_retrain_batch
                if since_last < self.cfg.retrain.cooldown_batches:
                    return Decision(
                        action=DecisionType.NONE,
                        reason="Retrain cooldown active",
                        metadata={
                            "cooldown_batches": self.cfg.retrain.cooldown_batches,
                            "batches_since_last_retrain": since_last,
                        },
                    )

            # Durable cooldown: persisted across restarts via action history.
            window = recent_actions[-self.cfg.retrain.cooldown_batches :]
            if DecisionType.RETRAIN in window:
                return Decision(
                    action=DecisionType.NONE,
                    reason="Recent retrain detected in action history",
                    metadata={"cooldown_batches": self.cfg.retrain.cooldown_batches},
                )

            return self._fire_retrain(
                batch_index,
                reason="Sustained performance degradation detected",
                metadata={
                    "f1_drop": f1_drop,
                    "baseline_f1": f1_baseline,
                    "current_f1": f1,
                    "trust_score": trust_score,
                },
            )

        # ------------------------------------------------------------------
        # 4. Stability → promote (hysteresis)
        # ------------------------------------------------------------------
        # Only fire when a staged candidate model exists.  Without this guard
        # the engine fires "promote" after N consecutive stable batches even
        # on a freshly-bootstrapped deployment that has never produced a
        # candidate, which causes the executor to raise FileNotFoundError.
        n = self.cfg.retrain.min_stable_batches
        if candidate_exists and len(recent_actions) >= n:
            window = recent_actions[-n:]
            if all(a == DecisionType.NONE for a in window):
                return Decision(
                    action=DecisionType.PROMOTE,
                    reason="Promotion stability conditions satisfied",
                    metadata={"stable_batches": n},
                )

        # ------------------------------------------------------------------
        # 5. Default → none
        # ------------------------------------------------------------------
        return Decision(
            action=DecisionType.NONE,
            reason="System operating within thresholds",
            metadata={"trust_score": trust_score},
        )
