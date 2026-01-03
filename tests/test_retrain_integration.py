import pandas as pd
import tempfile
from pathlib import Path

from model_monitor.training.retrain_pipeline import RetrainPipeline
from model_monitor.storage.model_store import ModelStore
from model_monitor.training.train import train_model


def test_retrain_pipeline_promotes_on_improvement(tmp_path: Path):
    """
    Full integration test:
    - trains an initial model
    - retrains on new data
    - promotes candidate if F1 improves
    """

    # --- Arrange ---
    model_store = ModelStore(base_path=tmp_path)

    # Initial training data
    baseline_df = pd.DataFrame({
        "feature_a": [0, 1, 0, 1],
        "feature_b": [1, 1, 0, 0],
        "label": [0, 1, 0, 1],
    })

    # Save an initial "current" model
    initial_model = train_model(baseline_df)
    model_store.save_candidate(initial_model)
    model_store.promote_candidate()

    # New data expected to improve performance
    retrain_df = pd.DataFrame({
        "feature_a": [0, 1, 1, 1, 0, 1],
        "feature_b": [1, 0, 1, 0, 0, 1],
        "label": [0, 1, 1, 1, 0, 1],
    })

    pipeline = RetrainPipeline()

    # --- Act ---
    result = pipeline.run(
        retrain_df,
        min_f1_improvement=0.01,
    )

    # --- Assert ---
    assert result.promotion.promoted is True
    assert result.promotion.candidate_f1 > result.promotion.current_f1

    # Ensure promoted model exists
    promoted_model = model_store.load_current()
    assert promoted_model is not None
