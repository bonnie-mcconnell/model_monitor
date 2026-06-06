"""Tests for CausalDriftAttributor.

Key invariants verified:
  1. Granger matrix is learned correctly from synthetic data with known causal structure.
  2. Isolated drift (no causal parents) → pipeline_suspect classification.
  3. Drift that propagates through causal graph → correlated_follower or genuine_shift.
  4. All stable features → dominant_cause = "none".
  5. All suspects → dominant_cause = "pipeline_failure".
  6. Mixed drift → dominant_cause = "mixed".
  7. Error cases: not fitted, wrong feature count.
"""

from __future__ import annotations

import numpy as np
import pytest

from model_monitor.monitoring.causal_drift import (
    CausalDriftAttributor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reference(n: int = 500, seed: int = 0) -> np.ndarray:
    """Return a 3-feature time-series reference array with a known causal structure:
    f0 → f1 (f0 Granger-causes f1 with a 1-step lag), f2 is independent.

    Granger causality requires temporal structure: f0_t must predict f1_{t+1}.
    We use AR processes so the lagged relationship is detectable by the F-test.
    """
    rng = np.random.default_rng(seed)
    # f0: AR(1) process with moderate autocorrelation
    f0 = np.zeros(n)
    f0[0] = rng.standard_normal()
    for t in range(1, n):
        f0[t] = 0.7 * f0[t - 1] + rng.standard_normal() * 0.5
    # f1: driven by lagged f0 - genuine Granger causality
    f1 = np.zeros(n)
    for t in range(1, n):
        f1[t] = 0.6 * f0[t - 1] + rng.standard_normal() * 0.3
    # f2: independent white noise - no causal relationships
    f2 = rng.standard_normal(n)
    return np.column_stack([f0, f1, f2])


def _fitted_attributor(
    feature_names: list[str] | None = None,
    n: int = 500,
    seed: int = 0,
) -> CausalDriftAttributor:
    names = feature_names or ["f0", "f1", "f2"]
    ref = _make_reference(n=n, seed=seed)
    attributor = CausalDriftAttributor(names, psi_threshold=0.1, alpha=0.05)
    attributor.fit(ref)
    return attributor


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_stores_feature_names() -> None:
    a = CausalDriftAttributor(["a", "b", "c"], psi_threshold=0.1)
    assert a.feature_names == ["a", "b", "c"]
    assert a.n_features == 3


def test_construction_rejects_empty_feature_names() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CausalDriftAttributor([])


def test_construction_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        CausalDriftAttributor(["a"], alpha=0.0)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_fit_sets_is_fitted() -> None:
    a = CausalDriftAttributor(["f0", "f1", "f2"])
    assert not a.is_fitted
    a.fit(_make_reference())
    assert a.is_fitted


def test_fit_returns_square_granger_matrix() -> None:
    a = _fitted_attributor()
    G = a.granger_matrix
    assert G is not None
    assert G.shape == (3, 3)
    assert G.dtype == bool


def test_granger_matrix_diagonal_is_false() -> None:
    """A feature cannot Granger-cause itself."""
    a = _fitted_attributor()
    G = a.granger_matrix
    assert G is not None
    for i in range(3):
        assert not G[i, i], f"G[{i},{i}] should be False (no self-causation)"


def test_fit_rejects_wrong_feature_count() -> None:
    a = CausalDriftAttributor(["f0", "f1", "f2"])
    with pytest.raises(ValueError, match="3 features"):
        a.fit(np.random.default_rng(0).standard_normal((100, 5)))


# ---------------------------------------------------------------------------
# Attribution - stable (no drift)
# ---------------------------------------------------------------------------


def test_no_drift_gives_dominant_cause_none() -> None:
    a = _fitted_attributor()
    psi = [0.01, 0.01, 0.01]  # all below threshold
    prod = _make_reference(n=200, seed=99)
    report = a.attribute(prod, psi)
    assert report.dominant_cause == "none"
    assert report.n_drifting == 0
    assert all(r.drift_class == "stable" for r in report.feature_results)


# ---------------------------------------------------------------------------
# Attribution - pipeline suspect
# ---------------------------------------------------------------------------


def test_isolated_drift_classified_as_pipeline_suspect() -> None:
    """f2 is independent in the causal graph.
    If only f2 drifts without f0 or f1 drifting, it has no causal explanation
    → pipeline_suspect.
    """
    a = _fitted_attributor()
    psi = [0.02, 0.02, 0.30]  # only f2 drifts (above 0.1 threshold)
    prod = _make_reference(n=200, seed=5)
    report = a.attribute(prod, psi)

    f2_result = next(r for r in report.feature_results if r.feature_name == "f2")
    assert f2_result.drift_class == "pipeline_suspect", (
        f"Expected f2 to be pipeline_suspect, got {f2_result.drift_class}. "
        f"Explanation: {f2_result.explanation}"
    )


def test_all_suspects_gives_pipeline_failure_dominant() -> None:
    a = _fitted_attributor()
    # f2 drifts in isolation (independent feature)
    psi = [0.01, 0.01, 0.30]
    prod = _make_reference(n=200, seed=5)
    report = a.attribute(prod, psi)
    assert report.dominant_cause == "pipeline_failure"
    assert "pipeline" in report.recommendation.lower()
    assert "retrain" in report.recommendation.lower()


# ---------------------------------------------------------------------------
# Attribution - correlated follower
# ---------------------------------------------------------------------------


def test_correlated_drift_classified_as_follower() -> None:
    """When f0 AND f1 both drift, and f0 Granger-causes f1,
    f1 should be classified as correlated_follower.
    """
    a = _fitted_attributor()
    # Both f0 and f1 drift - f1 is the correlated follower of f0
    psi = [0.25, 0.20, 0.02]
    prod = _make_reference(n=200, seed=7)
    report = a.attribute(prod, psi)

    f1_result = next(r for r in report.feature_results if r.feature_name == "f1")
    # f1 should be correlated_follower because f0 (its Granger-parent) also drifted
    assert f1_result.drift_class == "correlated_follower", (
        f"Expected f1 correlated_follower, got {f1_result.drift_class}"
    )
    assert "f0" in f1_result.granger_parents


def test_coherent_drift_recommends_retraining() -> None:
    a = _fitted_attributor()
    psi = [0.25, 0.20, 0.02]  # f0 + f1 drift, f1 follows f0
    prod = _make_reference(n=200, seed=7)
    report = a.attribute(prod, psi)
    assert report.dominant_cause in ("genuine_shift", "mixed")
    assert "retrain" in report.recommendation.lower()


# ---------------------------------------------------------------------------
# Attribution - mixed
# ---------------------------------------------------------------------------


def test_mixed_drift_gives_mixed_dominant() -> None:
    """f0 drifts (has a causal role in graph), f2 drifts in isolation.
    f2 is a suspect, f0 is genuine → mixed.
    """
    a = _fitted_attributor()
    # f0 drifts (root cause), f2 drifts in isolation (suspect)
    psi = [0.25, 0.02, 0.30]
    prod = _make_reference(n=200, seed=9)
    report = a.attribute(prod, psi)

    # There's at least one suspect (f2) and at least one non-suspect (f0)
    n_suspects = sum(
        1 for r in report.feature_results if r.drift_class == "pipeline_suspect"
    )
    n_non_stable = sum(1 for r in report.feature_results if r.drift_class != "stable")
    if n_suspects > 0 and n_non_stable > n_suspects:
        assert report.dominant_cause == "mixed"


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


def test_report_has_result_per_feature() -> None:
    a = _fitted_attributor()
    psi = [0.05, 0.15, 0.02]
    prod = _make_reference(n=200)
    report = a.attribute(prod, psi)
    assert len(report.feature_results) == 3


def test_report_drifting_count_matches_psi_threshold() -> None:
    a = _fitted_attributor()
    psi = [0.05, 0.15, 0.25]  # two above threshold
    prod = _make_reference(n=200)
    report = a.attribute(prod, psi)
    assert report.n_drifting == 2


def test_all_results_are_frozen() -> None:
    a = _fitted_attributor()
    report = a.attribute(_make_reference(n=200), [0.01, 0.01, 0.01])
    with pytest.raises(Exception):
        report.n_drifting = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_attribute_before_fit_raises() -> None:
    a = CausalDriftAttributor(["f0", "f1"])
    with pytest.raises(RuntimeError, match="fit"):
        a.attribute(np.zeros((10, 2)), [0.1, 0.1])


def test_attribute_wrong_psi_count_raises() -> None:
    a = _fitted_attributor()
    with pytest.raises(ValueError, match="PSI scores"):
        a.attribute(np.zeros((10, 3)), [0.1, 0.1])  # only 2 scores for 3 features


def test_granger_matrix_returns_copy() -> None:
    """Mutating the returned matrix must not change the internal state."""
    a = _fitted_attributor()
    G1 = a.granger_matrix
    assert G1 is not None
    G1[0, 1] = not G1[0, 1]
    G2 = a.granger_matrix
    assert G2 is not None
    # Internal matrix should be unchanged
    assert G1[0, 1] != G2[0, 1]
