from __future__ import annotations

import pytest

from model_monitor.config.settings import AppConfig, load_config
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import Decision


@pytest.fixture
def config() -> AppConfig:
    return load_config()


def test_no_action_when_system_healthy(config: AppConfig) -> None:
    engine = DecisionEngine(config)

    decision = engine.decide(
        batch_index=10,
        f1=0.90,
        f1_baseline=0.85,
        drift_score=0.01,
        trust_score=1.0,
    )

    assert isinstance(decision, Decision)
    assert decision.action == "none"


def test_retrain_triggered_on_f1_drop(config: AppConfig) -> None:
    engine = DecisionEngine(config)

    decision = engine.decide(
        batch_index=1,
        f1=0.80,
        f1_baseline=0.85,
        drift_score=0.01,
        trust_score=1.0,
    )

    assert decision.action == "retrain"
    assert "performance" in decision.reason.lower()


def test_retrain_respects_cooldown(config: AppConfig) -> None:
    engine = DecisionEngine(config)

    # First call establishes the retrain; batch_index=1 becomes _last_retrain_batch.
    # A second call within the cooldown window must be suppressed regardless of
    # how far F1 has dropped.
    engine.decide(
        batch_index=1,
        f1=0.80,
        f1_baseline=0.85,
        drift_score=0.01,
        trust_score=1.0,
    )

    decision = engine.decide(
        batch_index=2,
        f1=0.79,
        f1_baseline=0.85,
        drift_score=0.01,
        trust_score=1.0,
    )

    assert decision.action == "none"
    assert "cooldown" in decision.reason.lower()


def test_reject_on_severe_drift(config: AppConfig) -> None:
    engine = DecisionEngine(config)

    decision = engine.decide(
        batch_index=5,
        f1=0.90,
        f1_baseline=0.85,
        drift_score=config.drift.psi_threshold + 0.01,
        trust_score=1.0,
    )

    assert decision.action == "reject"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_invalid_trust_score_raises_value_error(config: AppConfig) -> None:
    """
    trust_score outside [0, 1] is a caller bug, not a monitoring signal.
    The engine must raise immediately rather than producing a misleading decision.
    """
    engine = DecisionEngine(config)
    with pytest.raises(ValueError, match="trust_score"):
        engine.decide(
            batch_index=1,
            trust_score=1.5,
            f1=0.85,
            f1_baseline=0.85,
            drift_score=0.0,
        )


def test_negative_f1_raises_value_error(config: AppConfig) -> None:
    engine = DecisionEngine(config)
    with pytest.raises(ValueError, match="f1"):
        engine.decide(
            batch_index=1,
            trust_score=0.9,
            f1=-0.1,
            f1_baseline=0.85,
            drift_score=0.0,
        )


def test_negative_f1_baseline_raises_value_error(config: AppConfig) -> None:
    engine = DecisionEngine(config)
    with pytest.raises(ValueError, match="f1_baseline"):
        engine.decide(
            batch_index=1,
            trust_score=0.9,
            f1=0.85,
            f1_baseline=-0.1,
            drift_score=0.0,
        )
