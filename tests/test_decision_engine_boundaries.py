from __future__ import annotations

from model_monitor.config.settings import load_config
from model_monitor.core.decision_engine import DecisionEngine


def test_retrain_vs_rollback_boundary() -> None:
    cfg = load_config()
    engine = DecisionEngine(cfg)

    baseline = 0.9

    # Just below rollback → retrain
    decision = engine.decide(
        batch_index=1,
        f1=baseline - (cfg.rollback.max_f1_drop - 0.01),
        f1_baseline=baseline,
        drift_score=0.0,
        trust_score=1.0,
    )
    assert decision.action == "retrain"

    # At rollback threshold → rollback
    decision = engine.decide(
        batch_index=2,
        f1=baseline - cfg.rollback.max_f1_drop,
        f1_baseline=baseline,
        drift_score=0.0,
        trust_score=1.0,
    )
    assert decision.action == "rollback"
