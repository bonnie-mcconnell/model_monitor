from __future__ import annotations

import numpy as np
import pandas as pd

from model_monitor.config.settings import load_config
from model_monitor.inference.predict import Predictor


class DummyModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), 2), 0.5)


def _make_batch(n: int = 50, n_features: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.random((n, n_features)), columns=[f"f{i}" for i in range(n_features)]
    )
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def test_predictor_healthy_batch_returns_none_decision() -> None:
    cfg = load_config()
    X, y = _make_batch()

    predictor = Predictor(config=cfg, f1_baseline=0.85)
    predictor.model = DummyModel()

    preds, confs, decision = predictor.predict_batch(X, y_true=y, batch_id="batch_001")

    assert len(preds) == 50
    assert len(confs) == 50
    assert decision.action == "none"


# ---------------------------------------------------------------------------
# Observability attributes: last_drift_score and last_trust_score
# ---------------------------------------------------------------------------


def test_last_drift_score_initialises_to_zero() -> None:
    """
    last_drift_score must start at 0.0 before any batch has been processed
    so callers can safely read it without first calling predict_batch.
    """
    cfg = load_config()
    predictor = Predictor(config=cfg)
    assert predictor.last_drift_score == 0.0


def test_last_trust_score_initialises_to_one() -> None:
    """
    last_trust_score starts at 1.0 (healthy) so callers that read before
    the first batch see a safe default rather than zero.
    """
    cfg = load_config()
    predictor = Predictor(config=cfg)
    assert predictor.last_trust_score == 1.0


def test_last_drift_score_set_after_predict_batch_without_baseline() -> None:
    """
    last_drift_score is set on every call even when f1_baseline is not
    provided (i.e. the decision engine is not triggered).

    This is the simulation loop use-case: drift monitoring runs regardless
    of whether the decision engine is active.
    """
    cfg = load_config()
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, (200, 3))

    X, _ = _make_batch()
    predictor = Predictor(config=cfg, reference_features=reference)
    predictor.model = DummyModel()

    # No f1_baseline - decision engine skipped, but drift still runs
    predictor.predict_batch(X, batch_id="no_baseline")

    # After one batch the DriftMonitor buffer isn't full yet (window=500),
    # so PSI is 0.0 - but the attribute must exist and be a float.
    assert isinstance(predictor.last_drift_score, float)


def test_last_trust_score_set_when_decision_engine_runs() -> None:
    """
    When f1_baseline is supplied and y_true is provided (batch_index > 1),
    last_trust_score must reflect the computed trust score, not the initial 1.0.
    A DummyModel with constant 0.5 confidence and random labels gives a
    trust score strictly below 1.0.
    """
    cfg = load_config()
    X, y = _make_batch()

    predictor = Predictor(config=cfg, f1_baseline=0.85)
    predictor.model = DummyModel()

    # batch_index must be > 1 for the engine to run; call twice
    predictor.predict_batch(X, y_true=y, batch_id="warm_up")
    predictor.predict_batch(X, y_true=y, batch_id="active")

    assert 0.0 <= predictor.last_trust_score <= 1.0
    # DummyModel with random labels should not produce a perfect trust score
    assert predictor.last_trust_score < 1.0


def test_last_drift_score_increases_after_distribution_shift() -> None:
    """
    When production data shifts away from the reference distribution,
    last_drift_score must increase as the DriftMonitor window fills.

    Verifies that the PSI signal actually flows through to the observable
    attribute - this is what the simulation loop relies on to display
    meaningful drift values.
    """
    from model_monitor.config.settings import (
        AppConfig,
        DriftConfig,
        RetrainConfig,
        RollbackConfig,
    )

    cfg = AppConfig(
        drift=DriftConfig(psi_threshold=0.2, window=3),  # small window for fast fill
        retrain=RetrainConfig(min_f1_gain=0.05, cooldown_batches=5, min_samples=100),
        rollback=RollbackConfig(max_f1_drop=0.15),
    )

    rng = np.random.default_rng(1)
    n_features = 4
    reference = rng.normal(0, 1, (500, n_features))

    predictor = Predictor(config=cfg, reference_features=reference)
    predictor.model = DummyModel()

    # Fill the window with shifted data (mean=5, far from reference mean=0)
    cols = [f"f{i}" for i in range(n_features)]
    for i in range(cfg.drift.window):
        X_shifted = pd.DataFrame(rng.normal(5, 1, (50, n_features)), columns=cols)
        predictor.predict_batch(X_shifted, batch_id=f"shifted_{i}")

    assert predictor.last_drift_score > 0.1, (
        f"Expected PSI > 0.1 after distribution shift, got {predictor.last_drift_score:.4f}"
    )
