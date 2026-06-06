"""
Tests for inference/shadow.py.

Key properties:
- Primary output is always returned regardless of candidate presence.
- Candidate runs silently without affecting primary state.
- ShadowStats accumulates correctly across batches.
- consume_shadow_stats() resets the counter.
- candidate_beats_primary reflects actual metric comparisons.
- Candidate failures are swallowed - primary still returns normally.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from model_monitor.config.settings import (
    AppConfig,
    DriftConfig,
    RetrainConfig,
    RollbackConfig,
)
from model_monitor.inference.predict import Predictor
from model_monitor.inference.shadow import (
    ShadowBatchResult,
    ShadowPredictor,
    ShadowStats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> AppConfig:
    return AppConfig(
        drift=DriftConfig(psi_threshold=0.2, window=5),
        retrain=RetrainConfig(min_f1_gain=0.02, cooldown_batches=5, min_samples=10),
        rollback=RollbackConfig(),
    )


def _make_predictor(
    base: Path, *, rng_seed: int = 42, f1_baseline: float = 0.80
) -> Predictor:
    """Build a real Predictor backed by a tiny trained RandomForest."""
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(rng_seed)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, models_dir / "current.pkl")
    (models_dir / "active.json").write_text(
        json.dumps({"version": "v1", "metrics": {"baseline_f1": f1_baseline}})
    )

    predictor = Predictor(
        config=_make_config(),
        model_path=models_dir / "current.pkl",
        active_file=models_dir / "active.json",
        f1_baseline=f1_baseline,
    )
    predictor.reload()
    return predictor


def _make_batch(n: int = 100, *, rng_seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(rng_seed)
    X = rng.normal(size=(n, 4))
    X_df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    y = pd.Series((X[:, 0] > 0).astype(int))
    return X_df, y


# ---------------------------------------------------------------------------
# Primary always returned
# ---------------------------------------------------------------------------


def test_primary_output_returned_without_candidate(tmp_path: Path) -> None:
    primary = _make_predictor(tmp_path / "p")
    shadow = ShadowPredictor(primary=primary)
    X, y = _make_batch()
    preds, confs, decision = shadow.predict_batch(X, y_true=y, batch_id="b0")
    assert len(preds) == 100
    assert len(confs) == 100


def test_primary_output_returned_with_candidate(tmp_path: Path) -> None:
    """ShadowPredictor must return exactly the primary's predictions."""
    primary = _make_predictor(tmp_path / "p", rng_seed=42)
    candidate = _make_predictor(tmp_path / "c", rng_seed=7)
    shadow = ShadowPredictor(primary=primary, candidate=candidate)

    X, y = _make_batch(rng_seed=5)
    shadow_preds, _, _ = shadow.predict_batch(X, y_true=y, batch_id="b0")

    # Reload primary independently and run the same batch to get ground truth.
    primary2 = _make_predictor(tmp_path / "p2", rng_seed=42)
    direct_preds, _, _ = primary2.predict_batch(X, y_true=y, batch_id="b1")

    np.testing.assert_array_equal(shadow_preds, direct_preds)


# ---------------------------------------------------------------------------
# ShadowStats accumulation
# ---------------------------------------------------------------------------


def test_shadow_stats_empty_before_any_batches(tmp_path: Path) -> None:
    primary = _make_predictor(tmp_path)
    shadow = ShadowPredictor(primary=primary)
    assert shadow.shadow_stats.n_batches == 0
    assert shadow.shadow_stats.total_samples == 0


def test_shadow_stats_accumulates_across_batches(tmp_path: Path) -> None:
    primary = _make_predictor(tmp_path / "p")
    candidate = _make_predictor(tmp_path / "c")
    shadow = ShadowPredictor(primary=primary, candidate=candidate)

    for i in range(3):
        X, y = _make_batch(rng_seed=i)
        shadow.predict_batch(X, y_true=y, batch_id=f"b{i}")

    assert shadow.shadow_stats.n_batches == 3
    assert shadow.shadow_stats.total_samples == 300


def test_consume_shadow_stats_resets_counter(tmp_path: Path) -> None:
    primary = _make_predictor(tmp_path / "p")
    candidate = _make_predictor(tmp_path / "c")
    shadow = ShadowPredictor(primary=primary, candidate=candidate)

    X, y = _make_batch()
    shadow.predict_batch(X, y_true=y, batch_id="b0")
    assert shadow.shadow_stats.n_batches == 1

    stats = shadow.consume_shadow_stats()
    assert stats.n_batches == 1
    assert shadow.shadow_stats.n_batches == 0


def test_reset_shadow_stats_clears_without_returning(tmp_path: Path) -> None:
    primary = _make_predictor(tmp_path / "p")
    candidate = _make_predictor(tmp_path / "c")
    shadow = ShadowPredictor(primary=primary, candidate=candidate)

    X, y = _make_batch()
    shadow.predict_batch(X, y_true=y, batch_id="b0")
    shadow.reset_shadow_stats()
    assert shadow.shadow_stats.n_batches == 0


# ---------------------------------------------------------------------------
# candidate_beats_primary logic (pure unit tests on ShadowStats)
# ---------------------------------------------------------------------------


def test_candidate_beats_primary_false_when_equal() -> None:
    stats = ShadowStats()
    result = ShadowBatchResult(
        batch_id="b0",
        primary_f1=0.80,
        candidate_f1=0.80,
        agreement_rate=1.0,
        primary_trust=0.80,
        candidate_trust=0.80,
        n_samples=100,
    )
    stats.update(result)
    assert not stats.candidate_beats_primary


def test_candidate_beats_primary_true_when_better() -> None:
    stats = ShadowStats()
    result = ShadowBatchResult(
        batch_id="b0",
        primary_f1=0.78,
        candidate_f1=0.84,
        agreement_rate=0.95,
        primary_trust=0.75,
        candidate_trust=0.82,
        n_samples=100,
    )
    stats.update(result)
    assert stats.candidate_beats_primary


def test_shadow_stats_mean_agreement_rate_correct() -> None:
    stats = ShadowStats()
    for agreement in [0.9, 0.8, 0.7]:
        stats.update(
            ShadowBatchResult(
                batch_id="b",
                primary_f1=0.8,
                candidate_f1=0.82,
                agreement_rate=agreement,
                primary_trust=0.8,
                candidate_trust=0.82,
                n_samples=100,
            )
        )
    assert abs(stats.mean_agreement_rate - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# Candidate failure is swallowed
# ---------------------------------------------------------------------------


def test_candidate_failure_does_not_raise(tmp_path: Path) -> None:
    """A crashing candidate must not propagate the exception to the caller."""
    primary = _make_predictor(tmp_path / "p")

    bad_candidate = MagicMock(spec=Predictor)
    bad_candidate.predict_batch.side_effect = RuntimeError("candidate exploded")

    shadow = ShadowPredictor(primary=primary, candidate=bad_candidate)

    X, y = _make_batch()
    # Must not raise - primary result returned normally.
    preds, confs, decision = shadow.predict_batch(X, y_true=y, batch_id="b0")
    assert len(preds) == 100


def test_shadow_stats_not_updated_on_candidate_failure(tmp_path: Path) -> None:
    """A failed candidate batch must not pollute the running stats."""
    primary = _make_predictor(tmp_path / "p")
    bad_candidate = MagicMock(spec=Predictor)
    bad_candidate.predict_batch.side_effect = RuntimeError("boom")
    shadow = ShadowPredictor(primary=primary, candidate=bad_candidate)

    X, y = _make_batch()
    shadow.predict_batch(X, y_true=y, batch_id="b0")
    assert shadow.shadow_stats.n_batches == 0


# ---------------------------------------------------------------------------
# has_candidate / set_candidate
# ---------------------------------------------------------------------------


def test_has_candidate_false_initially(tmp_path: Path) -> None:
    primary = _make_predictor(tmp_path)
    shadow = ShadowPredictor(primary=primary)
    assert not shadow.has_candidate()


def test_has_candidate_true_after_set(tmp_path: Path) -> None:
    primary = _make_predictor(tmp_path / "p")
    candidate = _make_predictor(tmp_path / "c")
    shadow = ShadowPredictor(primary=primary)
    shadow.set_candidate(candidate)
    assert shadow.has_candidate()


def test_set_candidate_none_clears_it(tmp_path: Path) -> None:
    primary = _make_predictor(tmp_path / "p")
    candidate = _make_predictor(tmp_path / "c")
    shadow = ShadowPredictor(primary=primary, candidate=candidate)
    shadow.set_candidate(None)
    assert not shadow.has_candidate()


def test_set_candidate_resets_stats(tmp_path: Path) -> None:
    """Swapping candidates resets accumulated stats to avoid stale comparisons."""
    primary = _make_predictor(tmp_path / "p")
    candidate = _make_predictor(tmp_path / "c")
    shadow = ShadowPredictor(primary=primary, candidate=candidate)

    X, y = _make_batch()
    shadow.predict_batch(X, y_true=y, batch_id="b0")
    assert shadow.shadow_stats.n_batches == 1

    shadow.set_candidate(candidate)  # swap to same candidate - stats must reset
    assert shadow.shadow_stats.n_batches == 0


# ---------------------------------------------------------------------------
# Passthrough properties
# ---------------------------------------------------------------------------


def test_last_metric_record_delegates_to_primary(tmp_path: Path) -> None:
    primary = _make_predictor(tmp_path)
    shadow = ShadowPredictor(primary=primary)
    X, y = _make_batch()
    shadow.predict_batch(X, y_true=y, batch_id="b0")
    assert shadow.last_metric_record is primary.last_metric_record


def test_last_drift_score_delegates_to_primary(tmp_path: Path) -> None:
    primary = _make_predictor(tmp_path)
    shadow = ShadowPredictor(primary=primary)
    X, y = _make_batch()
    shadow.predict_batch(X, y_true=y, batch_id="b0")
    assert shadow.last_drift_score == primary.last_drift_score


def test_last_trust_score_delegates_to_primary(tmp_path: Path) -> None:
    primary = _make_predictor(tmp_path)
    shadow = ShadowPredictor(primary=primary)
    X, y = _make_batch()
    shadow.predict_batch(X, y_true=y, batch_id="b0")
    assert shadow.last_trust_score == primary.last_trust_score
