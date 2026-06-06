"""Tests for ThresholdAdvisor - adaptive threshold calibration."""

from __future__ import annotations

import numpy as np
import pytest

from model_monitor.monitoring.threshold_advisor import (
    ThresholdAdvisor,
    ThresholdRecommendation,
)


def _stable_psi(n_features: int = 3, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(abs(rng.normal(0.03, 0.01))) for _ in range(n_features)]


def _fill_advisor(
    advisor: ThresholdAdvisor,
    n: int = 40,
    *,
    psi_mean: float = 0.04,
    trust_mean: float = 0.88,
) -> None:
    rng = np.random.default_rng(42)
    for i in range(n):
        psi = [
            max(0.0, float(rng.normal(psi_mean, 0.01)))
            for _ in range(advisor.n_features)
        ]
        trust = float(np.clip(rng.normal(trust_mean, 0.02), 0.0, 1.0))
        advisor.observe(psi, trust)


def test_construction_stores_feature_names() -> None:
    a = ThresholdAdvisor(["f0", "f1"], alpha=0.05)
    assert a.feature_names == ["f0", "f1"]


def test_construction_rejects_empty_features() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ThresholdAdvisor([])


def test_construction_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ThresholdAdvisor(["f0"], alpha=1.0)


def test_is_ready_false_before_min_batches() -> None:
    a = ThresholdAdvisor(["f0"], min_batches=30)
    for _ in range(29):
        a.observe([0.05], 0.88)
    assert not a.is_ready


def test_is_ready_true_at_min_batches() -> None:
    a = ThresholdAdvisor(["f0"], min_batches=30)
    for _ in range(30):
        a.observe([0.05], 0.88)
    assert a.is_ready


def test_recommend_raises_before_min_batches() -> None:
    a = ThresholdAdvisor(["f0", "f1"], min_batches=30)
    for _ in range(10):
        a.observe([0.05, 0.04], 0.88)
    with pytest.raises(ValueError, match="30"):
        a.recommend()


def test_recommend_returns_frozen_dataclass() -> None:
    a = ThresholdAdvisor(["f0", "f1", "f2"], min_batches=30)
    _fill_advisor(a)
    rec = a.recommend()
    assert isinstance(rec, ThresholdRecommendation)
    with pytest.raises(Exception):
        rec.psi_warn_global = 0.5  # type: ignore[misc]


def test_psi_thresholds_above_stable_mean() -> None:
    """Calibrated PSI threshold must exceed the mean of stable-period values."""
    a = ThresholdAdvisor(["f0"], min_batches=30, alpha=0.05)
    rng = np.random.default_rng(0)
    psi_vals = []
    for _ in range(50):
        psi = float(abs(rng.normal(0.04, 0.01)))
        psi_vals.append(psi)
        a.observe([psi], float(np.clip(rng.normal(0.88, 0.02), 0, 1)))
    rec = a.recommend()
    assert rec.psi_warn_per_feature[0] > float(np.mean(psi_vals))


def test_trust_warn_below_stable_mean() -> None:
    """Calibrated trust warn must be below the mean of stable-period trust scores."""
    a = ThresholdAdvisor(["f0"], min_batches=30, alpha=0.05)
    rng = np.random.default_rng(0)
    trust_vals = []
    for _ in range(50):
        trust = float(np.clip(rng.normal(0.88, 0.02), 0, 1))
        trust_vals.append(trust)
        a.observe([0.04], trust)
    rec = a.recommend()
    assert rec.trust_warn < float(np.mean(trust_vals))


def test_trust_critical_below_trust_warn() -> None:
    a = ThresholdAdvisor(["f0", "f1", "f2"], min_batches=30)
    _fill_advisor(a)
    rec = a.recommend()
    assert rec.trust_critical < rec.trust_warn


def test_n_observations_tracks_correctly() -> None:
    a = ThresholdAdvisor(["f0"], min_batches=10)
    assert a.n_observations == 0
    for i in range(15):
        a.observe([0.05], 0.88)
    assert a.n_observations == 15


def test_feature_names_in_recommendation() -> None:
    a = ThresholdAdvisor(["age", "income", "score"], min_batches=30)
    _fill_advisor(a, n=40)
    rec = a.recommend()
    assert rec.feature_names == ("age", "income", "score")
    assert len(rec.psi_warn_per_feature) == 3


def test_high_variance_feature_generates_note() -> None:
    """A naturally high-variance feature should generate a warning note."""
    a = ThresholdAdvisor(["volatile", "stable"], min_batches=30, alpha=0.05)
    rng = np.random.default_rng(0)
    for _ in range(50):
        volatile = float(abs(rng.normal(0.15, 0.08)))  # high mean AND high variance
        stable_v = float(abs(rng.normal(0.03, 0.005)))
        a.observe([volatile, stable_v], float(np.clip(rng.normal(0.88, 0.02), 0, 1)))
    rec = a.recommend()
    assert any("volatile" in note for note in rec.notes), (
        f"Expected note about volatile feature, got: {rec.notes}"
    )


def test_observe_rejects_wrong_psi_count() -> None:
    a = ThresholdAdvisor(["f0", "f1"])
    with pytest.raises(ValueError, match="2"):
        a.observe([0.05, 0.04, 0.03], 0.88)


def test_observe_rejects_out_of_range_trust() -> None:
    a = ThresholdAdvisor(["f0"])
    with pytest.raises(ValueError, match="trust_score"):
        a.observe([0.05], 1.5)


def test_psi_thresholds_respect_alpha() -> None:
    """At alpha=0.10, approx 10% of stable-period PSI should exceed threshold."""
    a = ThresholdAdvisor(["f0"], min_batches=30, alpha=0.10)
    rng = np.random.default_rng(1)
    psi_vals = []
    for _ in range(200):
        psi = float(abs(rng.normal(0.05, 0.02)))
        psi_vals.append(psi)
        a.observe([psi], float(np.clip(rng.normal(0.88, 0.02), 0, 1)))
    rec = a.recommend()
    exceedance_rate = sum(1 for p in psi_vals if p > rec.psi_warn_per_feature[0]) / len(
        psi_vals
    )
    # Should be close to alpha=0.10 (within 5pp on 200 samples)
    assert abs(exceedance_rate - 0.10) < 0.07, (
        f"Exceedance rate {exceedance_rate:.3f} too far from alpha=0.10"
    )
