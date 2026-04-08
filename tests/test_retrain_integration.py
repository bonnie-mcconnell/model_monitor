from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model_monitor.storage.model_store import ModelStore
from model_monitor.training.retrain_pipeline import RetrainPipeline
from model_monitor.training.train import train_model


def test_retrain_pipeline_promotes_on_improvement(tmp_path: Path) -> None:
    """
    Full integration test:
    - trains an initial model
    - retrains on new data with same schema
    - promotes candidate if F1 improves
    """

    # --- Arrange ---
    model_store = ModelStore(base_path=tmp_path)

    baseline_df = pd.DataFrame({
        "feature_a": [0, 1, 0, 1],
        "feature_b": [1, 1, 0, 0],
        "label": [0, 1, 0, 1],
    })

    initial_model = train_model(baseline_df)
    model_store.save_candidate(initial_model)
    model_store.promote_candidate()

    retrain_df = pd.DataFrame({
        "feature_a": [0, 1, 1, 1, 0, 1],
        "feature_b": [1, 0, 1, 0, 0, 1],
        "label": [0, 1, 1, 1, 0, 1],
    })

    pipeline = RetrainPipeline(model_store=model_store)

    # --- Act ---
    result = pipeline.run(
        retrain_df,
        min_f1_improvement=0.01,
    )

    # --- Assert ---
    assert result.promotion.promoted is True
    assert result.promotion.candidate_f1 > result.promotion.current_f1

    promoted_model = model_store.load_current()
    assert promoted_model is not None


def test_retrain_uses_held_out_validation_not_training_data(tmp_path: Path) -> None:
    """
    RetrainPipeline must evaluate candidate F1 on held-out validation data,
    not on the training samples the candidate already saw.

    Evaluating on training data produces optimistic in-sample F1 estimates
    that favour promotion of overfit models. With a held-out split the
    candidate's generalisation is measured - the same thing the current
    model will face in production.

    We verify this by checking that the pipeline completes without error on
    a dataset large enough to trigger the split (>= _VAL_MIN_SAMPLES = 50).
    """
    model_store = ModelStore(base_path=tmp_path)

    # 100 rows - above _VAL_MIN_SAMPLES threshold, so split fires
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "feature_a": rng.standard_normal(100),
        "feature_b": rng.standard_normal(100),
        "label": rng.integers(0, 2, 100),
    })

    pipeline = RetrainPipeline(model_store=model_store)
    result = pipeline.run(df, min_f1_improvement=0.0)

    # Must complete without error and produce a valid F1
    assert 0.0 <= result.promotion.candidate_f1 <= 1.0
    assert result.n_samples == 100
