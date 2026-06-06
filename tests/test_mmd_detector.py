"""Tests for the MMD multivariate drift detector (monitoring.mmd).

These tests verify the statistical correctness of the detector, not just
its interface.  Specifically:

  - Under the null hypothesis (identical distributions), the empirical
    rejection rate must be approximately alpha.
  - Under a clear alternative (shifted distributions), the detector must
    have near-100% power.
  - The detector correctly identifies joint drift that PSI *cannot* detect
    - the primary reason for its existence.
  - Interface contracts: wrong shapes raise, degenerate inputs are handled.
"""

from __future__ import annotations

import numpy as np
import pytest

from model_monitor.monitoring.mmd import (
    MMDDriftDetector,
    MMDDriftResult,
    _mmd2_unbiased,
    _rbf_kernel,
    median_bandwidth,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _standard_ref(n: int = 500, d: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


# ---------------------------------------------------------------------------
# median_bandwidth
# ---------------------------------------------------------------------------


class TestMedianBandwidth:
    def test_positive_on_standard_normal(self) -> None:
        X = _standard_ref(200, 4)
        assert median_bandwidth(X) > 0.0

    def test_raises_on_1d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            median_bandwidth(np.ones(10))

    def test_raises_on_too_few_rows(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            median_bandwidth(np.ones((1, 4)))

    def test_constant_input_returns_one(self) -> None:
        """Degenerate all-constant input falls back to bandwidth=1.0."""
        X = np.ones((50, 3))
        assert median_bandwidth(X) == 1.0

    def test_pooled_bandwidth_uses_both_arrays(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 4))
        Y = rng.standard_normal((100, 4)) * 5.0  # very different scale
        bw_X_only = median_bandwidth(X)
        bw_pooled = median_bandwidth(X, Y)
        # Pooling Y (larger scale) must increase the bandwidth estimate.
        assert bw_pooled > bw_X_only


# ---------------------------------------------------------------------------
# _rbf_kernel
# ---------------------------------------------------------------------------


class TestRbfKernel:
    def test_diagonal_is_one_for_same_matrix(self) -> None:
        """k(x, x) = 1 for all x when the two input matrices are the same array."""
        X = _standard_ref(20, 4)
        K = _rbf_kernel(X, X, sigma=1.0)
        assert np.allclose(np.diag(K), 1.0)

    def test_values_in_unit_interval(self) -> None:
        X = _standard_ref(30, 4)
        Y = _standard_ref(30, 4, seed=1)
        K = _rbf_kernel(X, Y, sigma=1.0)
        assert K.min() >= 0.0
        assert K.max() <= 1.0 + 1e-9

    def test_output_shape(self) -> None:
        X = _standard_ref(20, 4)
        Y = _standard_ref(30, 4, seed=1)
        K = _rbf_kernel(X, Y, sigma=1.0)
        assert K.shape == (20, 30)

    def test_large_bandwidth_approaches_one(self) -> None:
        """σ → ∞ makes all kernel values approach 1."""
        X = _standard_ref(20, 4)
        Y = _standard_ref(20, 4, seed=1)
        K = _rbf_kernel(X, Y, sigma=1e9)
        assert np.allclose(K, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# _mmd2_unbiased
# ---------------------------------------------------------------------------


class TestMMD2Unbiased:
    def test_near_zero_for_identical_distributions(self) -> None:
        """MMD² ≈ 0 when both samples are drawn from the same distribution."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 4))
        Y = rng.standard_normal((300, 4))
        sigma = median_bandwidth(X, Y)
        mmd2 = _mmd2_unbiased(X, Y, sigma)
        # Unbiased estimator is zero in expectation; small fluctuations are fine.
        assert abs(mmd2) < 0.05

    def test_large_for_shifted_distributions(self) -> None:
        """MMD² should be clearly positive for mean-shifted samples."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((300, 4))
        Y = rng.standard_normal((300, 4)) + 3.0  # strong mean shift
        sigma = median_bandwidth(X, Y)
        mmd2 = _mmd2_unbiased(X, Y, sigma)
        assert mmd2 > 0.1


# ---------------------------------------------------------------------------
# MMDDriftDetector - construction
# ---------------------------------------------------------------------------


class TestMMDDriftDetectorConstruction:
    def test_raises_on_1d_reference(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            MMDDriftDetector(np.ones(100))

    def test_raises_on_too_few_reference_rows(self) -> None:
        with pytest.raises(ValueError, match="at least 10"):
            MMDDriftDetector(np.ones((5, 4)))

    def test_raises_on_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            MMDDriftDetector(_standard_ref(), alpha=0.0)

    def test_raises_on_too_few_permutations(self) -> None:
        with pytest.raises(ValueError, match="n_permutations"):
            MMDDriftDetector(_standard_ref(), n_permutations=5)

    def test_raises_on_non_positive_bandwidth(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MMDDriftDetector(_standard_ref(), bandwidth=0.0)

    def test_bandwidth_inferred_from_reference(self) -> None:
        ref = _standard_ref()
        det = MMDDriftDetector(ref)
        assert det.bandwidth > 0.0

    def test_explicit_bandwidth_respected(self) -> None:
        ref = _standard_ref()
        det = MMDDriftDetector(ref, bandwidth=2.5)
        assert det.bandwidth == pytest.approx(2.5)

    def test_n_reference_property(self) -> None:
        ref = _standard_ref(400)
        det = MMDDriftDetector(ref)
        assert det.n_reference == 400


# ---------------------------------------------------------------------------
# MMDDriftDetector.test()
# ---------------------------------------------------------------------------


class TestMMDDriftDetectorTest:
    def test_raises_on_1d_production(self) -> None:
        det = MMDDriftDetector(_standard_ref())
        with pytest.raises(ValueError, match="2D"):
            det.test(np.ones(50))

    def test_raises_on_wrong_n_features(self) -> None:
        det = MMDDriftDetector(_standard_ref(200, d=4))
        with pytest.raises(ValueError, match="reference has 4"):
            det.test(_standard_ref(100, d=6))

    def test_raises_on_too_few_production_rows(self) -> None:
        det = MMDDriftDetector(_standard_ref())
        with pytest.raises(ValueError, match="at least 5"):
            det.test(_standard_ref(3))

    def test_returns_mmd_drift_result(self) -> None:
        det = MMDDriftDetector(_standard_ref())
        result = det.test(_standard_ref(100, seed=99))
        assert isinstance(result, MMDDriftResult)
        assert result.drift_type == "joint"

    def test_p_value_in_unit_interval(self) -> None:
        det = MMDDriftDetector(_standard_ref())
        result = det.test(_standard_ref(100, seed=5))
        assert 0.0 < result.p_value <= 1.0

    def test_mmd2_near_zero_for_same_distribution(self) -> None:
        """Under H₀ the MMD² statistic should be small."""
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((500, 6))
        prod = rng.standard_normal((300, 6))
        det = MMDDriftDetector(ref, n_permutations=100, random_state=0)
        result = det.test(prod)
        # Unbiased MMD² has mean zero under H₀; a few SDs away is unusual.
        assert abs(result.mmd2) < 0.2

    def test_is_drift_true_for_strong_mean_shift(self) -> None:
        """Clearly shifted production data should always trigger drift."""
        rng = np.random.default_rng(3)
        ref = rng.standard_normal((500, 6))
        prod = rng.standard_normal((300, 6)) + 4.0  # 4 sigma mean shift
        det = MMDDriftDetector(ref, n_permutations=200, random_state=3)
        result = det.test(prod)
        assert result.is_drift
        assert result.p_value < 0.05

    def test_is_drift_false_for_null_distribution(self) -> None:
        """Under H₀ with large samples, p-value should exceed alpha."""
        rng = np.random.default_rng(11)
        ref = rng.standard_normal((600, 5))
        prod = rng.standard_normal((400, 5))
        det = MMDDriftDetector(ref, n_permutations=200, random_state=11)
        result = det.test(prod)
        # This is stochastic - we use a very lenient threshold to avoid flakiness.
        # The key property is that p is not vanishingly small under H₀.
        assert result.p_value > 0.01

    def test_detects_joint_drift_invisible_to_marginals(self) -> None:
        """Core use case: rotation of joint distribution with stable marginals.

        We rotate a 2D Gaussian by 90 degrees - both marginals remain
        N(0, 1) but the joint distribution changes completely (correlation
        flips from +0.9 to −0.9).  PSI-based detectors would see zero
        marginal drift.  MMD must detect the joint shift.
        """
        rng = np.random.default_rng(42)
        n = 600

        # Reference: correlated Gaussian (ρ = 0.9)
        cov_ref = np.array([[1.0, 0.9], [0.9, 1.0]])
        ref = rng.multivariate_normal([0.0, 0.0], cov_ref, size=n)

        # Production: anti-correlated Gaussian (ρ = −0.9)
        # Both marginals are still N(0, 1) - PSI would see zero drift.
        cov_prod = np.array([[1.0, -0.9], [-0.9, 1.0]])
        prod = rng.multivariate_normal([0.0, 0.0], cov_prod, size=n // 2)

        det = MMDDriftDetector(ref, n_permutations=200, random_state=42)
        result = det.test(prod)

        assert result.is_drift, (
            "MMD failed to detect joint drift (correlation sign flip) "
            f"that is invisible to marginal tests. p={result.p_value:.4f}"
        )

    def test_result_fields_consistent(self) -> None:
        """is_drift is the logical implication of p_value < alpha."""
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((300, 4))
        prod = rng.standard_normal((200, 4)) + 3.0
        det = MMDDriftDetector(ref, alpha=0.05, n_permutations=100, random_state=0)
        result = det.test(prod)
        assert result.is_drift == (result.p_value < result.alpha)
        assert result.alpha == 0.05
        assert result.n_reference > 0
        assert result.n_production > 0
        assert result.bandwidth == det.bandwidth


# ---------------------------------------------------------------------------
# Null hypothesis size test  (statistical calibration)
# ---------------------------------------------------------------------------


class TestNullHypothesisCalibration:
    """Verify that the rejection rate under H₀ is approximately alpha.

    We run the test 40 times with independent samples from the same
    distribution and count rejections.  The expected count is 40 * alpha.
    We allow a ±3 sigma binomial tolerance to avoid flakiness while
    still catching a badly miscalibrated p-value.
    """

    def test_rejection_rate_approximately_alpha(self) -> None:
        alpha = 0.05
        n_trials = 40
        n_permutations = 200

        rng = np.random.default_rng(999)
        ref = rng.standard_normal((400, 4))
        det = MMDDriftDetector(
            ref, alpha=alpha, n_permutations=n_permutations, random_state=0
        )

        rejections = 0
        for _ in range(n_trials):
            prod = rng.standard_normal((200, 4))
            if det.test(prod).is_drift:
                rejections += 1

        expected = n_trials * alpha  # 2.0
        # 3-sigma binomial tolerance: σ = sqrt(n*p*(1-p)) ≈ 1.38
        tolerance = 3.0 * (n_trials * alpha * (1 - alpha)) ** 0.5
        assert abs(rejections - expected) <= tolerance, (
            f"Rejection rate {rejections}/{n_trials} deviates more than 3σ "
            f"from expected {expected} (tolerance ±{tolerance:.1f})"
        )
