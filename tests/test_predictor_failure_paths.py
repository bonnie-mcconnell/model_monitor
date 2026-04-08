from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_monitor.config.settings import load_config
from model_monitor.inference.predict import Predictor


class DummyModel:
    """Always predicts class 0 with certainty - produces F1=0.0 against any y=1 labels."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 1.0
        return probs


def test_predictor_returns_none_decision_before_second_batch() -> None:
    """
    The decision engine requires at least two batches (batch_index > 1) to
    avoid making decisions on insufficient signal. The first call must always
    return action="none" with reason="insufficient_signal_for_decision".
    """
    cfg = load_config()
    predictor = Predictor(config=cfg, f1_baseline=0.9)
    predictor.model = DummyModel()

    X = pd.DataFrame(np.random.rand(50, 3), columns=["a", "b", "c"])
    y = pd.Series(np.zeros(50))

    _, _, decision = predictor.predict_batch(X, y_true=y, batch_id="first_batch")

    assert decision.action == "none"
    assert "insufficient_signal" in decision.reason


def test_predictor_returns_none_decision_without_labels() -> None:
    """
    When y_true is not provided the engine cannot compute F1 - it must
    return action="none" rather than silently using a stale or zero F1.
    """
    cfg = load_config()
    predictor = Predictor(config=cfg, f1_baseline=0.9)
    predictor.model = DummyModel()

    X = pd.DataFrame(np.random.rand(50, 3), columns=["a", "b", "c"])

    _, _, decision = predictor.predict_batch(X, batch_id="no_labels")

    assert decision.action == "none"


def test_predictor_triggers_retrain_on_catastrophic_f1_drop() -> None:
    """
    DummyModel always predicts class 0. With y=ones (all class 1), the
    F1 score is 0.0 - a catastrophic drop from baseline 0.9.
    After the second batch (when the engine is active), the decision must
    be retrain or rollback, never none or reject.
    """
    cfg = load_config()
    predictor = Predictor(config=cfg, f1_baseline=0.9)
    predictor.model = DummyModel()

    X = pd.DataFrame(np.random.rand(100, 3), columns=["a", "b", "c"])
    # All labels are class 1; DummyModel always predicts class 0 → F1 = 0.0
    y = pd.Series(np.ones(100))

    # First batch - engine skipped (batch_index == 1)
    predictor.predict_batch(X, y_true=y, batch_id="warmup")

    # Second batch - engine now active
    _, _, decision = predictor.predict_batch(X, y_true=y, batch_id="active")

    assert decision.action in {"retrain", "rollback"}, (
        f"Expected retrain or rollback for F1=0.0 vs baseline=0.9, "
        f"got action={decision.action!r} reason={decision.reason!r}"
    )


def test_predictor_raises_on_non_dataframe_input() -> None:
    """predict_batch must reject non-DataFrame input with a clear TypeError."""
    cfg = load_config()
    predictor = Predictor(config=cfg, f1_baseline=0.9)
    predictor.model = DummyModel()

    with pytest.raises(TypeError, match="DataFrame"):
        predictor.predict_batch(
            np.random.rand(10, 3),  # noqa: PD901 - intentionally wrong type to test TypeError
            batch_id="bad_input",
        )

