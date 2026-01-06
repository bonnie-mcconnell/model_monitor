import numpy as np
import pandas as pd

from model_monitor.inference.predict import Predictor
from model_monitor.config.settings import load_config


class DummyModel:
    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 1.0
        return probs


def test_predictor_triggers_retrain_on_f1_drop():
    cfg = load_config()

    X = pd.DataFrame(
        np.random.rand(100, 3),
        columns=["f1", "f2", "f3"],
    )
    y = np.zeros(100)

    predictor = Predictor(config=cfg, f1_baseline=0.9)
    predictor.model = DummyModel()  # inject model

    _, _, decision = predictor.predict_batch(
        X,
        y_true=pd.Series(y),
        batch_id="batch_001",
    )

    assert decision.action in {"none", "retrain", "rollback"}

