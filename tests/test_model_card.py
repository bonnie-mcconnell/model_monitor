"""Tests for the model card / training provenance system (training.model_card).

Covers:
  - build_model_card: correct field population
  - hash_training_data: determinism and sensitivity
  - feature_schema_from_array: dtype, bounds, null_rate
  - ModelCard.to_json / from_json round-trip
  - ModelCard.save / load file round-trip
  - summary_dict shape and values
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from model_monitor.training.model_card import (
    ModelCard,
    ModelEvaluation,
    build_model_card,
    feature_schema_from_array,
    hash_training_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def eval_metrics() -> ModelEvaluation:
    return ModelEvaluation(
        accuracy=0.91,
        f1=0.88,
        f1_improvement=0.023,
        n_eval_samples=200,
        bootstrap_ci_lower=0.005,
        bootstrap_ci_upper=0.041,
    )


@pytest.fixture()
def sample_card(eval_metrics: ModelEvaluation) -> ModelCard:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 4))
    return build_model_card(
        model_version=3,
        X_train=X,
        feature_names=["age", "income", "score", "tenure"],
        evaluation=eval_metrics,
        promotion_reason="f1_improvement=0.023",
        pipeline_version="1.2.0",
    )


# ---------------------------------------------------------------------------
# hash_training_data
# ---------------------------------------------------------------------------


class TestHashTrainingData:
    def test_deterministic(self) -> None:
        X = np.random.default_rng(0).standard_normal((200, 4))
        assert hash_training_data(X) == hash_training_data(X)

    def test_64_hex_chars(self) -> None:
        X = np.ones((10, 3))
        h = hash_training_data(X)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_data_gives_different_hash(self) -> None:
        rng = np.random.default_rng(0)
        X1 = rng.standard_normal((100, 4))
        X2 = rng.standard_normal((100, 4))
        assert hash_training_data(X1) != hash_training_data(X2)

    def test_hash_is_over_values_not_shape(self) -> None:
        """The hash encodes raw float64 bytes - shape is not included.

        This matches RetrainEvidenceBuffer's SHA-256 strategy: two arrays with
        identical values but different shapes will produce the same hash.  This
        is an accepted limitation and is consistent with the deduplication key.
        """
        X1 = np.arange(12, dtype=float).reshape(4, 3)
        X2 = np.arange(12, dtype=float).reshape(3, 4)
        # Same bytes → same hash (documented behaviour).
        assert hash_training_data(X1) == hash_training_data(X2)

    def test_accepts_dataframe(self) -> None:
        import pandas as pd

        X = np.ones((10, 3))
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        h_arr = hash_training_data(X)
        h_df = hash_training_data(df)
        assert h_arr == h_df


# ---------------------------------------------------------------------------
# feature_schema_from_array
# ---------------------------------------------------------------------------


class TestFeatureSchemaFromArray:
    def test_correct_count(self) -> None:
        X = np.ones((50, 5))
        specs = feature_schema_from_array(X)
        assert len(specs) == 5

    def test_auto_names_f0_to_fn(self) -> None:
        X = np.ones((50, 3))
        specs = feature_schema_from_array(X)
        assert [s.name for s in specs] == ["f0", "f1", "f2"]

    def test_explicit_names_respected(self) -> None:
        X = np.ones((50, 3))
        specs = feature_schema_from_array(X, feature_names=["a", "b", "c"])
        assert [s.name for s in specs] == ["a", "b", "c"]

    def test_min_max_bounds(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 2))
        specs = feature_schema_from_array(X)
        assert specs[0].min_value is not None
        assert specs[0].max_value is not None
        assert specs[0].min_value < specs[0].max_value

    def test_null_rate_zero_for_clean_data(self) -> None:
        X = np.ones((100, 3))
        specs = feature_schema_from_array(X)
        assert all(s.null_rate == 0.0 for s in specs)

    def test_null_rate_non_zero_for_nan_data(self) -> None:
        X = np.ones((100, 2))
        X[:20, 0] = np.nan
        specs = feature_schema_from_array(X)
        assert abs(specs[0].null_rate - 0.20) < 0.01
        assert specs[1].null_rate == 0.0

    def test_dataframe_infers_column_names(self) -> None:
        import pandas as pd

        X = np.ones((50, 3))
        df = pd.DataFrame(X, columns=["alpha", "beta", "gamma"])
        specs = feature_schema_from_array(df)
        assert [s.name for s in specs] == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# build_model_card
# ---------------------------------------------------------------------------


class TestBuildModelCard:
    def test_version_set(self, sample_card: ModelCard) -> None:
        assert sample_card.model_version == 3

    def test_feature_names_set(self, sample_card: ModelCard) -> None:
        assert [f.name for f in sample_card.feature_schema] == [
            "age", "income", "score", "tenure"
        ]

    def test_n_training_samples_correct(self, sample_card: ModelCard) -> None:
        assert sample_card.n_training_samples == 500

    def test_training_hash_non_empty(self, sample_card: ModelCard) -> None:
        assert len(sample_card.training_data_hash) == 64

    def test_evaluation_metrics_set(
        self, sample_card: ModelCard, eval_metrics: ModelEvaluation
    ) -> None:
        assert sample_card.evaluation.f1 == pytest.approx(eval_metrics.f1)
        assert sample_card.evaluation.f1_improvement == pytest.approx(
            eval_metrics.f1_improvement
        )

    def test_pipeline_version_set(self, sample_card: ModelCard) -> None:
        assert sample_card.pipeline_version == "1.2.0"

    def test_promotion_reason_set(self, sample_card: ModelCard) -> None:
        assert "0.023" in sample_card.promotion_reason

    def test_created_at_is_recent(self, sample_card: ModelCard) -> None:
        import time

        assert abs(sample_card.created_at - time.time()) < 5

    def test_created_at_iso_is_string(self, sample_card: ModelCard) -> None:
        assert "T" in sample_card.created_at_iso  # ISO 8601

    def test_extra_metadata_stored(self, eval_metrics: ModelEvaluation) -> None:
        X = np.ones((50, 2))
        card = build_model_card(
            model_version=1,
            X_train=X,
            evaluation=eval_metrics,
            extra={"experiment_id": "exp_42", "n_estimators": 100},
        )
        assert card.extra["experiment_id"] == "exp_42"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestModelCardSerialisation:
    def test_to_json_from_json_round_trip(self, sample_card: ModelCard) -> None:
        restored = ModelCard.from_json(sample_card.to_json())
        assert restored.model_version == sample_card.model_version
        assert restored.training_data_hash == sample_card.training_data_hash
        assert restored.evaluation.f1 == pytest.approx(sample_card.evaluation.f1)
        assert len(restored.feature_schema) == len(sample_card.feature_schema)
        assert restored.feature_schema[0].name == sample_card.feature_schema[0].name

    def test_save_load_file_round_trip(
        self, sample_card: ModelCard, tmp_path: Path
    ) -> None:
        path = tmp_path / "cards" / "v3_card.json"
        sample_card.save(path)
        assert path.exists()
        restored = ModelCard.load(path)
        assert restored.model_version == sample_card.model_version
        assert restored.pipeline_version == sample_card.pipeline_version

    def test_json_is_valid(self, sample_card: ModelCard) -> None:
        import json

        data = json.loads(sample_card.to_json())
        assert isinstance(data, dict)
        assert "model_version" in data
        assert "feature_schema" in data
        assert "evaluation" in data

    def test_summary_dict_keys(self, sample_card: ModelCard) -> None:
        s = sample_card.summary_dict()
        required = {
            "model_version", "created_at_iso", "training_data_hash",
            "n_training_samples", "n_features", "feature_names",
            "accuracy", "f1", "f1_improvement", "promotion_reason",
        }
        assert required <= set(s.keys())

    def test_summary_dict_n_features(self, sample_card: ModelCard) -> None:
        assert sample_card.summary_dict()["n_features"] == 4


# ---------------------------------------------------------------------------
# Hash consistency with RetrainEvidenceBuffer
# ---------------------------------------------------------------------------


class TestHashConsistencyWithRetrain:
    """hash_training_data must produce the same hash as RetrainEvidenceBuffer.

    Both use SHA-256 over float64 bytes.  This test uses the same approach
    directly so the card hash is comparable to the retrain deduplication key.
    """

    def test_hash_matches_manual_sha256(self) -> None:
        import hashlib

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        expected = hashlib.sha256(X.astype(np.float64).tobytes()).hexdigest()
        assert hash_training_data(X) == expected
