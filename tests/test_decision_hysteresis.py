from __future__ import annotations

from typing import cast

import pytest

from model_monitor.config.settings import (
    AppConfig,
    DriftConfig,
    RetrainConfig,
    RollbackConfig,
)
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import DecisionType


@pytest.fixture
def decision_engine() -> DecisionEngine:
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


def test_retrain_blocked_by_batch_cooldown(decision_engine: DecisionEngine) -> None:
    decision_engine._last_retrain_batch = 10

    decision = decision_engine.decide(
        batch_index=12,
        trust_score=0.8,
        f1=0.7,
        f1_baseline=0.8,
        drift_score=0.0,
        recent_actions=[],
    )

    assert decision.action == DecisionType.NONE
    assert "cooldown" in decision.reason.lower()


def test_retrain_blocked_by_recent_actions(decision_engine: DecisionEngine) -> None:
    decision = decision_engine.decide(
        batch_index=20,
        trust_score=0.8,
        f1=0.7,
        f1_baseline=0.8,
        drift_score=0.0,
        recent_actions=[DecisionType.RETRAIN, DecisionType.NONE, DecisionType.NONE],
    )

    assert decision.action == DecisionType.NONE
    assert "recent" in decision.reason.lower()


def test_promote_requires_stable_none_window(decision_engine: DecisionEngine) -> None:
    decision = decision_engine.decide(
        batch_index=30,
        trust_score=0.9,
        f1=0.85,
        f1_baseline=0.85,
        drift_score=0.0,
        recent_actions=[DecisionType.NONE, DecisionType.NONE, DecisionType.RETRAIN, DecisionType.NONE, DecisionType.NONE],
    )

    assert decision.action != DecisionType.PROMOTE


# ---------------------------------------------------------------------------
# Dual-cooldown disagreement: the two cooldown mechanisms are independent
# and can disagree.  Both must be respected - the engine blocks retrain
# when *either* cooldown is active.
# ---------------------------------------------------------------------------


def test_ephemeral_cooldown_blocks_when_durable_cleared(
    decision_engine: DecisionEngine,
) -> None:
    """
    Ephemeral cooldown (batch-index distance) still active, durable
    cooldown (recent_actions window) already cleared.

    The engine must honour the ephemeral block and return "none".
    This tests the case where the process has not restarted since the
    last retrain, so _last_retrain_batch is still set, but enough
    "none" actions have accumulated in the store to clear the durable
    window check.
    """
    # Retrain fired at batch 10; cooldown_batches=5 means block until >15.
    decision_engine._last_retrain_batch = 10

    # Durable window is clear: last cooldown_batches (5) actions are all none.
    recent = cast(
        list[DecisionType], ["retrain", "none", "none", "none", "none", "none"]
    )

    decision = decision_engine.decide(
        batch_index=13,  # only 3 batches since last retrain, still in window
        trust_score=0.8,
        f1=0.7,
        f1_baseline=0.8,
        drift_score=0.0,
        recent_actions=recent,
    )

    assert decision.action == DecisionType.NONE
    # Ephemeral cooldown fires first in the priority chain
    assert "cooldown" in decision.reason.lower()


def test_durable_cooldown_blocks_when_ephemeral_cleared(
    decision_engine: DecisionEngine,
) -> None:
    """
    Durable cooldown (recent_actions) still active, ephemeral cooldown
    (batch-index distance) has cleared.

    The engine must honour the durable block and return "none".
    This tests the case where the process restarted after the last
    retrain (so _last_retrain_batch is None), but the DecisionStore
    still shows a recent retrain within the cooldown window.
    """
    # Simulate a process restart: ephemeral state is gone.
    decision_engine._last_retrain_batch = None

    # Durable window is NOT clear: "retrain" appears in last 5 actions.
    recent = cast(list[DecisionType], ["none", "retrain", "none", "none", "none"])

    decision = decision_engine.decide(
        batch_index=999,  # far from any ephemeral reference
        trust_score=0.8,
        f1=0.7,
        f1_baseline=0.8,
        drift_score=0.0,
        recent_actions=recent,
    )

    assert decision.action == DecisionType.NONE
    assert "recent" in decision.reason.lower()


def test_retrain_fires_when_both_cooldowns_cleared(
    decision_engine: DecisionEngine,
) -> None:
    """
    When both cooldowns have cleared, a degraded model should trigger
    retrain.  This is the positive control: the cooldown tests above
    only prove suppression; this test proves the engine is not stuck.
    """
    decision_engine._last_retrain_batch = None  # ephemeral cleared

    # Durable window: retrain is older than cooldown_batches (5) actions.
    recent = cast(
        list[DecisionType], ["retrain", "none", "none", "none", "none", "none"]
    )

    decision = decision_engine.decide(
        batch_index=50,
        trust_score=0.8,
        f1=0.7,
        f1_baseline=0.8,
        drift_score=0.0,
        recent_actions=recent,
    )

    assert decision.action == "retrain"
