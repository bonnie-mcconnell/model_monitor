from __future__ import annotations

import numpy as np
import pandas as pd

from model_monitor.config.settings import load_config
from model_monitor.inference.predict import Predictor


class DummyModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), 2), 0.5)


def test_predictor_healthy_batch_returns_none_decision() -> None:
    cfg = load_config()

    X = pd.DataFrame(
        np.random.rand(50, 3),
        columns=["a", "b", "c"],
    )
    y = np.random.randint(0, 2, size=50)

    predictor = Predictor(config=cfg, f1_baseline=0.85)
    predictor.model = DummyModel()

    preds, confs, decision = predictor.predict_batch(
        X,
        y_true=pd.Series(y),
        batch_id="batch_healthy_001",
    )

    assert len(preds) == 50
    assert len(confs) == 50
    assert decision.action == "none"
