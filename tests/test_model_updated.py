from pathlib import Path

import numpy as np

from model_monitor.inference.predict import Predictor
from model_monitor.storage.model_store import ModelStore
from model_monitor.config.settings import AppConfig, RetrainConfig, DriftConfig, load_config


class DummyModel:
    """
    Minimal deterministic model for testing.
    """
    def __init__(self, value: int):
        self.value = value

    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, self.value] = 1.0
        return probs


def test_predictor_reads_model_version_from_active_json(tmp_path: Path):
    """
    When a model is promoted, model metadata should be written to active.json,
    and the Predictor should be able to read the active model version.
    """

    store = ModelStore(base_path=tmp_path)

    cfg = load_config()

    predictor = Predictor(
        config=cfg,
        model_path=store.current,
        active_file=store.active_file
    )

    model = DummyModel(value=1)
    store.save_candidate(model)
    store.promote_candidate(metrics={"f1": 0.82})

    version = predictor.current_model_version()

    assert version is not None
    assert isinstance(version, str)
    assert len(version) > 0
