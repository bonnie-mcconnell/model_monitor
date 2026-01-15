def test_retrain_not_triggered_during_cooldown(decision_engine):
    decision_engine._cooldown_remaining = 3

    decision = decision_engine.evaluate(
        trust_score=0.2,
        f1_drop=0.4,
        has_labels=True,
    )

    assert decision.action == "none"
    assert "cooldown" in decision.reason.lower()
