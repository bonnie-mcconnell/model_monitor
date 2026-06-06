"""Tests for the retrain circuit breaker in DecisionEngine.

The circuit breaker prevents infinite retrain loops when every retrain produces
a model that still drifts.  These tests verify:
  - Normal operation is unaffected while under the cap
  - SYSTEM_ERROR is returned exactly at the cap
  - The error message is informative
  - reset_retrain_counter() re-opens the breaker
  - cap=0 disables the breaker entirely
  - Both retrain paths (sustained degradation and reject-escalation) share the same
    counter and are both subject to the cap
"""

from __future__ import annotations

from model_monitor.config.settings import (
    AppConfig,
    DriftConfig,
    RetrainConfig,
    RollbackConfig,
)
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import DecisionType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(max_retrain_attempts: int = 3) -> AppConfig:
    return AppConfig(
        drift=DriftConfig(psi_threshold=0.25, window=3),
        retrain=RetrainConfig(
            min_f1_gain=0.05,
            cooldown_batches=1,  # short cooldown so tests stay concise
            min_samples=10,
            evidence_window=1,
            min_stable_batches=2,
            max_retrain_attempts=max_retrain_attempts,
        ),
        rollback=RollbackConfig(max_f1_drop=0.20),
    )


def _trigger_retrain(engine: DecisionEngine, batch_index: int) -> None:
    """Advance the engine through cooldown and trigger a retrain decision."""
    # Burn the cooldown by issuing batches with no degradation first,
    # then one degraded batch.
    engine._last_retrain_batch = None  # reset ephemeral cooldown
    engine.decide(
        batch_index=batch_index,
        trust_score=0.5,
        f1=0.73,  # drop=0.07: above min_f1_gain=0.05, below max_f1_drop=0.20
        f1_baseline=0.80,
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )


# ---------------------------------------------------------------------------
# Normal operation - under the cap
# ---------------------------------------------------------------------------


def test_first_retrain_fires_normally() -> None:
    engine = DecisionEngine(_config(max_retrain_attempts=5))
    decision = engine.decide(
        batch_index=10,
        trust_score=0.5,
        f1=0.73,
        f1_baseline=0.80,
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )
    assert decision.action == DecisionType.RETRAIN
    assert engine.retrain_attempt_count == 1


def test_counter_increments_with_each_retrain() -> None:
    engine = DecisionEngine(_config(max_retrain_attempts=10))
    for i in range(3):
        engine._last_retrain_batch = None  # bypass ephemeral cooldown
        engine.decide(
            batch_index=i * 10,
            trust_score=0.5,
            f1=0.73,
            f1_baseline=0.80,
            drift_score=0.05,
            recent_actions=[DecisionType.NONE],
        )
    assert engine.retrain_attempt_count == 3


# ---------------------------------------------------------------------------
# Circuit breaker fires at cap
# ---------------------------------------------------------------------------


def test_circuit_breaker_fires_at_cap() -> None:
    """After max_retrain_attempts retrains the engine must return SYSTEM_ERROR."""
    cap = 3
    engine = DecisionEngine(_config(max_retrain_attempts=cap))

    # Exhaust the cap
    for i in range(cap):
        engine._last_retrain_batch = None
        d = engine.decide(
            batch_index=i * 5,
            trust_score=0.5,
            f1=0.73,
            f1_baseline=0.80,
            drift_score=0.05,
            recent_actions=[DecisionType.NONE],
        )
        assert d.action == DecisionType.RETRAIN, f"Expected RETRAIN on attempt {i + 1}"

    assert engine.retrain_attempt_count == cap

    # Next retrain trigger must return SYSTEM_ERROR
    engine._last_retrain_batch = None
    d_err = engine.decide(
        batch_index=cap * 5,
        trust_score=0.5,
        f1=0.73,
        f1_baseline=0.80,
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )
    assert d_err.action == DecisionType.SYSTEM_ERROR


def test_circuit_breaker_error_message_contains_counts() -> None:
    """SYSTEM_ERROR reason must tell the operator what happened and how to fix it."""
    cap = 2
    engine = DecisionEngine(_config(max_retrain_attempts=cap))
    for i in range(cap):
        engine._last_retrain_batch = None
        engine.decide(
            batch_index=i * 5,
            trust_score=0.5,
            f1=0.73,
            f1_baseline=0.80,
            drift_score=0.05,
            recent_actions=[DecisionType.NONE],
        )

    engine._last_retrain_batch = None
    d = engine.decide(
        batch_index=cap * 5,
        trust_score=0.5,
        f1=0.73,
        f1_baseline=0.80,
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )
    assert str(cap) in d.reason
    assert "reset_retrain_counter" in d.reason
    assert d.metadata.get("max_retrain_attempts") == cap


def test_circuit_breaker_metadata_has_attempt_count() -> None:
    cap = 2
    engine = DecisionEngine(_config(max_retrain_attempts=cap))
    for i in range(cap):
        engine._last_retrain_batch = None
        engine.decide(
            batch_index=i * 5,
            trust_score=0.5,
            f1=0.73,
            f1_baseline=0.80,
            drift_score=0.05,
            recent_actions=[DecisionType.NONE],
        )
    engine._last_retrain_batch = None
    d = engine.decide(
        batch_index=cap * 5,
        trust_score=0.5,
        f1=0.73,
        f1_baseline=0.80,
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )
    assert d.metadata.get("retrain_attempt_count") == cap


def test_system_error_persists_until_reset() -> None:
    """Once the breaker fires it should stay open on subsequent decisions."""
    cap = 1
    engine = DecisionEngine(_config(max_retrain_attempts=cap))
    engine.decide(
        batch_index=0,
        trust_score=0.5,
        f1=0.73,
        f1_baseline=0.80,
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )
    # Now every retrain-triggering decision should return SYSTEM_ERROR
    for i in range(3):
        engine._last_retrain_batch = None
        d = engine.decide(
            batch_index=(i + 1) * 5,
            trust_score=0.5,
            f1=0.73,
            f1_baseline=0.80,
            drift_score=0.05,
            recent_actions=[DecisionType.NONE],
        )
        assert d.action == DecisionType.SYSTEM_ERROR


# ---------------------------------------------------------------------------
# Reset re-opens the breaker
# ---------------------------------------------------------------------------


def test_reset_allows_retrains_again() -> None:
    cap = 2
    engine = DecisionEngine(_config(max_retrain_attempts=cap))

    # Exhaust cap
    for i in range(cap):
        engine._last_retrain_batch = None
        engine.decide(
            batch_index=i * 5,
            trust_score=0.5,
            f1=0.73,
            f1_baseline=0.80,
            drift_score=0.05,
            recent_actions=[DecisionType.NONE],
        )

    # Verify breaker is open
    engine._last_retrain_batch = None
    d_err = engine.decide(
        batch_index=99,
        trust_score=0.5,
        f1=0.73,
        f1_baseline=0.80,
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )
    assert d_err.action == DecisionType.SYSTEM_ERROR

    # Reset
    engine.reset_retrain_counter()
    assert engine.retrain_attempt_count == 0

    # Should retrain again
    d_ok = engine.decide(
        batch_index=100,
        trust_score=0.5,
        f1=0.73,
        f1_baseline=0.80,
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )
    assert d_ok.action == DecisionType.RETRAIN
    assert engine.retrain_attempt_count == 1


def test_reset_clears_ephemeral_cooldown_too() -> None:
    engine = DecisionEngine(_config(max_retrain_attempts=5))
    engine._last_retrain_batch = 50  # set a recent retrain batch
    engine.reset_retrain_counter()
    assert engine._last_retrain_batch is None


# ---------------------------------------------------------------------------
# Cap = 0 disables circuit breaker
# ---------------------------------------------------------------------------


def test_zero_cap_disables_circuit_breaker() -> None:
    """max_retrain_attempts=0 means unlimited retrains - breaker is disabled."""
    engine = DecisionEngine(_config(max_retrain_attempts=0))
    for i in range(20):
        engine._last_retrain_batch = None
        d = engine.decide(
            batch_index=i * 5,
            trust_score=0.5,
            f1=0.73,
            f1_baseline=0.80,
            drift_score=0.05,
            recent_actions=[DecisionType.NONE],
        )
        assert d.action == DecisionType.RETRAIN, (
            f"Expected RETRAIN on attempt {i + 1} with cap=0, got {d.action}"
        )


# ---------------------------------------------------------------------------
# Reject-escalation path also counts toward cap
# ---------------------------------------------------------------------------


def test_reject_escalation_retrain_counts_toward_cap() -> None:
    """Retrains triggered by reject-escalation increment the same counter."""
    cap = 2
    engine = DecisionEngine(_config(max_retrain_attempts=cap))
    n = engine.cfg.retrain.cooldown_batches

    # Trigger retrain via sustained-degradation path
    engine.decide(
        batch_index=0,
        trust_score=0.5,
        f1=0.73,
        f1_baseline=0.80,
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )
    assert engine.retrain_attempt_count == 1

    # Now trigger via reject-escalation path (fill reject window)
    engine._last_retrain_batch = None
    reject_history = [DecisionType.REJECT] * n
    d = engine.decide(
        batch_index=50,
        trust_score=0.3,
        f1=0.80,
        f1_baseline=0.80,
        drift_score=0.30,  # above psi_threshold=0.25
        recent_actions=reject_history,
    )
    assert d.action == DecisionType.RETRAIN
    assert engine.retrain_attempt_count == cap

    # Cap reached - next escalation should return SYSTEM_ERROR
    engine._last_retrain_batch = None
    d_err = engine.decide(
        batch_index=100,
        trust_score=0.3,
        f1=0.80,
        f1_baseline=0.80,
        drift_score=0.30,
        recent_actions=reject_history,
    )
    assert d_err.action == DecisionType.SYSTEM_ERROR


# ---------------------------------------------------------------------------
# Non-retrain decisions are unaffected by circuit breaker
# ---------------------------------------------------------------------------


def test_none_decision_unaffected_by_circuit_breaker() -> None:
    """With breaker open, a healthy batch should still return NONE."""
    cap = 1
    engine = DecisionEngine(_config(max_retrain_attempts=cap))
    # Open the breaker
    engine.decide(
        batch_index=0,
        trust_score=0.5,
        f1=0.73,
        f1_baseline=0.80,
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )

    # Healthy batch - should still return NONE, not SYSTEM_ERROR
    d = engine.decide(
        batch_index=10,
        trust_score=0.95,
        f1=0.90,
        f1_baseline=0.80,
        drift_score=0.02,
        recent_actions=[DecisionType.NONE],
    )
    assert d.action == DecisionType.NONE


def test_rollback_decision_unaffected_by_circuit_breaker() -> None:
    """Rollback is independent of the retrain counter."""
    cap = 1
    engine = DecisionEngine(_config(max_retrain_attempts=cap))
    # Open the breaker
    engine.decide(
        batch_index=0,
        trust_score=0.5,
        f1=0.73,
        f1_baseline=0.80,
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )

    # Catastrophic regression should still trigger rollback
    d = engine.decide(
        batch_index=20,
        trust_score=0.1,
        f1=0.50,
        f1_baseline=0.80,  # 0.30 drop > max_f1_drop=0.20
        drift_score=0.05,
        recent_actions=[DecisionType.NONE],
    )
    assert d.action == DecisionType.ROLLBACK
