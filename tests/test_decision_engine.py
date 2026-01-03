import pytest
from pathlib import Path

from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import Decision
from model_monitor.config.settings import load_config


@pytest.fixture
def config():
    return load_config()


def test_no_action_when_system_healthy(config):
    engine = DecisionEngine(config)

    decision = engine.decide(
        batch_index=10,
        f1=0.90,
        f1_baseline=0.85,
        drift_score=0.01,
    )

    assert isinstance(decision, Decision)
    assert decision.action == "none"


def test_retrain_triggered_on_f1_drop(config):
    engine = DecisionEngine(config)

    decision = engine.decide(
        batch_index=1,
        f1=0.80,
        f1_baseline=0.85,
        drift_score=0.01,
    )

    assert decision.action == "retrain"
    assert "performance" in decision.reason.lower()


def test_retrain_respects_cooldown(config):
    engine = DecisionEngine(config)

    # First retrain
    engine.decide(
        batch_index=1,
        f1=0.80,
        f1_baseline=0.85,
        drift_score=0.01,
    )

    # Immediately again
    decision = engine.decide(
        batch_index=2,
        f1=0.79,
        f1_baseline=0.85,
        drift_score=0.01,
    )

    assert decision.action == "none"
    assert "cooldown" in decision.reason.lower()


def test_reject_on_severe_drift(config):
    engine = DecisionEngine(config)

    decision = engine.decide(
        batch_index=5,
        f1=0.90,
        f1_baseline=0.85,
        drift_score=config.drift.psi_threshold + 0.01,
    )

    assert decision.action == "reject"
