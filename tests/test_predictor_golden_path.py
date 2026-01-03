import numpy as np
import pandas as pd
from pathlib import Path

from model_monitor.inference.predict import Predictor
from model_monitor.config.settings import load_config


def test_predictor_healthy_batch():
    cfg = load_config()

    X = pd.DataFrame(
        np.random.rand(50, 3),
        columns=["f1", "f2", "f3"],
    )
    y = np.random.randint(0, 2, size=50)

    predictor = Predictor(
        config=cfg,
        f1_baseline=0.85,
    )

    preds, confs, decision = predictor.predict_batch(
        X,
        y_true=pd.Series(y),
        batch_id="batch_healthy_001",
    )

    assert len(preds) == 50
    assert len(confs) == 50
    assert decision.action == "none"
