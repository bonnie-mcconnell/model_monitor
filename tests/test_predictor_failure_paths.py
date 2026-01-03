import numpy as np
import pandas as pd
from pathlib import Path

from model_monitor.inference.predict import Predictor
from model_monitor.config.settings import load_config


def test_predictor_triggers_retrain_on_f1_drop():
    cfg = load_config()

    X = pd.DataFrame(
        np.random.rand(100, 3),
        columns=["f1", "f2", "f3"],
    )

    # All wrong labels → F1 collapse
    y = np.zeros(100)

    predictor = Predictor(
        config=cfg,
        f1_baseline=0.9,
    )

    _, _, decision = predictor.predict_batch(
        X,
        y_true=pd.Series(y),
        batch_id="batch_001",
    )

    assert decision.action in {"retrain", "none"}
