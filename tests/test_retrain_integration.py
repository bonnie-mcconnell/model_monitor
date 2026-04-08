from __future__ import annotations

from pathlib import Path

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
