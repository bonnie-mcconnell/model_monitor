import time
import numpy as np
from pathlib import Path

from model_monitor.inference.predict import Predictor
from model_monitor.storage.model_store import ModelStore
from model_monitor.config.settings import load_config


class DummyModel:
    def __init__(self, value: int):
        self.value = value

    def predict_proba(self, X):
        probs = np.zeros((len(X), 3))
        probs[:, self.value] = 1.0
        return probs


def test_predictor_reloads_after_promotion(tmp_path: Path):
    store = ModelStore(base_path=tmp_path)
    cfg = load_config()

    predictor = Predictor(
        config=cfg,
        model_path=store.current,
        active_file=store.active_file,
    )

    model_v1 = DummyModel(value=1)
    store.save_candidate(model_v1)
    store.promote_candidate(metrics={"f1": 0.6})

    assert predictor.reload_if_changed() is True
    preds_v1 = predictor.active_model.predict_proba([0, 0, 0]).argmax(axis=1)
    assert preds_v1.tolist() == [1, 1, 1]

    time.sleep(1)

    model_v2 = DummyModel(value=2)
    store.save_candidate(model_v2)
    store.promote_candidate(metrics={"f1": 0.7})

    # Promotion changes version → reload expected
    assert predictor.reload_if_changed() is True
    preds_v2 = predictor.active_model.predict_proba([0, 0]).argmax(axis=1)
    assert preds_v2.tolist() == [2, 2]


def test_predictor_reload_noop_when_model_unchanged(tmp_path: Path):
    store = ModelStore(base_path=tmp_path)
    cfg = load_config()

    model = DummyModel(value=0)
    store.save_candidate(model)
    store.promote_candidate(metrics={"f1": 0.9})

    predictor = Predictor(
        config=cfg,
        model_path=store.current,
        active_file=store.active_file,
    )

    assert predictor.reload_if_changed() is True
    assert predictor.reload_if_changed() is False
