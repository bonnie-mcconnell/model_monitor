import pytest

from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.config.settings import (
    AppConfig,
    RetrainConfig,
    RollbackConfig,
    DriftConfig,
)


@pytest.fixture
def decision_engine():
    cfg = AppConfig(
        retrain=RetrainConfig(
            min_f1_gain=0.05,
            cooldown_batches=5,
            min_stable_batches=3,
            min_samples=1000,
        ),
        rollback=RollbackConfig(
            max_f1_drop=0.25,
        ),
        drift=DriftConfig(
            psi_threshold=0.3,
            window=50,
        ),
    )
    return DecisionEngine(cfg)


def test_retrain_blocked_by_batch_cooldown(decision_engine):
    decision_engine._last_retrain_batch = 10

    decision = decision_engine.decide(
        batch_index=12,
        trust_score=0.8,
        f1=0.7,
        f1_baseline=0.8,
        drift_score=0.0,
        recent_actions=[],
    )

    assert decision.action == "none"
    assert "cooldown" in decision.reason.lower()


def test_retrain_blocked_by_recent_actions(decision_engine):
    decision = decision_engine.decide(
        batch_index=20,
        trust_score=0.8,
        f1=0.7,
        f1_baseline=0.8,
        drift_score=0.0,
        recent_actions=["retrain", "none", "none"],
    )

    assert decision.action == "none"
    assert "recent" in decision.reason.lower()


def test_promote_requires_stable_none_window(decision_engine):
    decision = decision_engine.decide(
        batch_index=30,
        trust_score=0.9,
        f1=0.85,
        f1_baseline=0.85,
        drift_score=0.0,
        recent_actions=["none", "none", "retrain", "none", "none"],
    )

    assert decision.action != "promote"
