import time
from pathlib import Path

import numpy as np

from model_monitor.inference.predict import Predictor
from model_monitor.storage.model_store import ModelStore
from model_monitor.config.settings import AppConfig, RetrainConfig, DriftConfig, load_config


class DummyModel:
    def __init__(self, value: int):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_predictor_reloads_after_promotion(tmp_path: Path):
    # --- Setup isolated model store ---
    store = ModelStore(base_path=tmp_path)

    # --- Minimal valid config (matches real schema) ---
    cfg = load_config()

    # --- Predictor pointing at temp model path ---
    predictor = Predictor(
        config=cfg,
        model_path=store.current,
        active_file=store.active_file
    )

    # --- First model ---
    model_v1 = DummyModel(value=1)
    store.save_candidate(model_v1)
    store.promote_candidate(metrics={"f1": 0.6})

    assert predictor.reload_if_changed() is True
    preds_v1 = predictor.active_model.predict([0, 0, 0])
    assert preds_v1.tolist() == [1, 1, 1]

    # Ensure filesystem mtime changes
    time.sleep(1)

    # --- Second model ---
    model_v2 = DummyModel(value=2)
    store.save_candidate(model_v2)
    store.promote_candidate(metrics={"f1": 0.7})

    assert predictor.reload_if_changed() is False
    preds_v2 = predictor.active_model.predict([0, 0])
    assert preds_v2.tolist() == [2, 2]

def test_predictor_reload_noop_when_model_unchanged(tmp_path: Path):
    store = ModelStore(base_path=tmp_path)

    cfg = AppConfig(
        retrain=RetrainConfig(
            min_f1_gain=0.0,
            min_samples=1,
            cooldown_batches=0,
        ),
        drift=DriftConfig(
            psi_threshold=1.0,
            window=1,
        ),
    )

    model = DummyModel(value=42)
    store.save_candidate(model)
    store.promote_candidate(metrics={"f1": 0.9})

    predictor = Predictor(
        config=cfg,
        model_path=store.current,
    )

    # First call loads model and records mtime
    assert predictor.reload_if_changed() is True

    # Second call should NOOP
    assert predictor.reload_if_changed() is False
