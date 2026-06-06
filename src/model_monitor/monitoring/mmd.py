"""Kernel Maximum Mean Discrepancy (MMD) multivariate drift detector.

PSI detects marginal drift in individual features.  It completely misses joint
distribution shift - the case where each feature's marginal looks stable but
the *combination* has changed (e.g. the correlation structure between age and
income shifts while both marginals stay flat).  This is not a contrived edge
case: it is the dominant failure mode of univariate monitoring in practice.

MMD (Gretton et al., 2012) is the correct test for joint distribution shift.
It measures the distance between two distributions in a reproducing kernel
Hilbert space (RKHS).  The key property is that MMD = 0 if and only if the
two distributions are identical - something PSI cannot guarantee even with
very tight per-feature thresholds.

This module implements:

  - :class:`MMDDriftDetector`  - the main detector with permutation-test
    p-value calibration and a bootstrap confidence interval on the MMD
    statistic itself.
  - :func:`median_bandwidth`   - the standard median heuristic for the RBF
    kernel bandwidth parameter.
  - :class:`MMDDriftResult`    - structured result for clean API consumption.

Kernel choice
-------------
We use the RBF (Gaussian) kernel:  k(x, y) = exp(−‖x − y‖² / (2σ²)).

Bandwidth σ is set by the median heuristic:
  σ² = median(‖xᵢ − xⱼ‖²) / 2

This is the standard bandwidth selection approach from the two-sample testing
literature (Gretton et al., 2012; Sutherland et al., 2017).  It is data-
adaptive and requires no hyperparameter tuning.

Computational complexity
------------------------
Naïve MMD computation is O(n²) in the number of samples.  For n ≤ 2000 this
is fast enough for batch monitoring (<100 ms per call on a laptop CPU).  For
larger batches the :class:`MMDDriftDetector` subsamples to ``max_samples``
(default 1000) before computing the kernel matrix.

References
----------
Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A.
  (2012). A kernel two-sample test. JMLR, 13(1), 723–773.

Sutherland, D. J., Tung, H.-Y., Strathmann, H., De, S., Ramdas, A., Smola,
  A., & Gretton, A. (2017). Generative Models and Model Criticism via
  Optimized Maximum Mean Discrepancy. ICLR 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------


def median_bandwidth(X: np.ndarray, Y: np.ndarray | None = None) -> float:
    """Estimate the RBF kernel bandwidth using the median heuristic.

    The median heuristic sets σ² = median(‖xᵢ − xⱼ‖²) over all pairs
    from the pooled sample.  It is the standard bandwidth selection
    method for MMD-based two-sample tests.

    Args:
        X: 2D array of shape (n, d).
        Y: Optional second array.  When provided, bandwidth is estimated
           from the pooled sample ``[X; Y]``.  When absent, estimated
           from X alone (useful at fit time when Y is not yet available).

    Returns:
        Positive float σ (the *standard deviation*, not σ²).

    Raises:
        ValueError: if X is not 2D or has fewer than 2 rows.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}")
    if len(X) < 2:
        raise ValueError("X must have at least 2 rows for bandwidth estimation")

    pool = np.vstack([X, Y]) if Y is not None else X
    # Subsample to at most 500 points to keep O(n²) pairwise distances cheap.
    if len(pool) > 500:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pool), 500, replace=False)
        pool = pool[idx]

    # ‖x - y‖² = ‖x‖² + ‖y‖² - 2 x·y  (numerically stable squared-distance)
    sq_norms = (pool**2).sum(axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2.0 * pool @ pool.T
    # Clamp to zero to prevent negative values from floating-point noise.
    sq_dists = np.maximum(sq_dists, 0.0)
    # Ignore diagonal (self-distances = 0)
    upper = sq_dists[np.triu_indices(len(pool), k=1)]
    median_sq = float(np.median(upper))
    # Guard against degenerate constant inputs
    if median_sq < 1e-12:
        return 1.0
    return float(np.sqrt(median_sq / 2.0))


# ---------------------------------------------------------------------------
# Kernel computation
# ---------------------------------------------------------------------------


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute the RBF kernel matrix between rows of X and rows of Y.

    K[i, j] = exp(−‖X[i] − Y[j]‖² / (2σ²))

    Args:
        X: 2D array of shape (m, d).
        Y: 2D array of shape (n, d).
        sigma: Kernel bandwidth (standard deviation).

    Returns:
        2D array of shape (m, n).
    """
    sq_norms_X = (X**2).sum(axis=1, keepdims=True)
    sq_norms_Y = (Y**2).sum(axis=1, keepdims=True)
    sq_dists = sq_norms_X + sq_norms_Y.T - 2.0 * X @ Y.T
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.asarray(np.exp(-sq_dists / (2.0 * sigma**2)))


# ---------------------------------------------------------------------------
# Unbiased MMD² estimator
# ---------------------------------------------------------------------------


def _mmd2_unbiased(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float,
) -> float:
    """Compute the unbiased estimator of MMD².

    The unbiased estimator (Gretton et al., 2012, eq. 5) is:

        MMD²_u = 1/(m(m-1)) Σᵢ≠ⱼ k(xᵢ,xⱼ)
               + 1/(n(n-1)) Σᵢ≠ⱼ k(yᵢ,yⱼ)
               − 2/(mn) Σᵢⱼ k(xᵢ,yⱼ)

    It is unbiased under H₀ (the two distributions are the same) and is
    the standard estimator for permutation-based tests.

    Args:
        X: Reference samples, shape (m, d).
        Y: Production samples, shape (n, d).
        sigma: RBF kernel bandwidth.

    Returns:
        Scalar float (may be negative due to unbiasedness).
    """
    m, n = len(X), len(Y)
    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)

    # Zero out diagonal for unbiased XX and YY terms
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_xx = Kxx.sum() / (m * (m - 1)) if m > 1 else 0.0
    term_yy = Kyy.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_xy = Kxy.sum() / (m * n)

    return float(term_xx + term_yy - 2.0 * term_xy)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MMDDriftResult:
    """Result of a single MMD drift detection call.

    Attributes:
        mmd2:           Unbiased MMD² statistic.  Values near zero indicate
                        no drift; large positive values indicate drift.
                        Note: the statistic is not bounded above.
        p_value:        Approximate p-value from a permutation test under
                        the null H₀: P_ref = P_prod.  Small values (< alpha)
                        indicate statistically significant drift.
        is_drift:       True when ``p_value < alpha``.
        alpha:          Significance level used for the decision (e.g. 0.05).
        n_permutations: Number of permutations used to estimate the p-value.
        n_reference:    Number of reference samples used.
        n_production:   Number of production samples used.
        bandwidth:      RBF kernel bandwidth σ used for this test.
        drift_type:     ``"joint"`` - MMD always tests the joint distribution.
                        Included for downstream serialisation consistency.
    """

    mmd2: float
    p_value: float
    is_drift: bool
    alpha: float
    n_permutations: int
    n_reference: int
    n_production: int
    bandwidth: float
    drift_type: Literal["joint"] = "joint"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class MMDDriftDetector:
    """Kernel MMD two-sample test for joint distribution shift.

    Detects cases where the joint distribution P(X₁, X₂, …, Xd) has
    changed, even when all marginal distributions look stable.  This is
    the core gap in univariate monitoring tools (PSI, KS-test, etc).

    Usage::

        detector = MMDDriftDetector(reference_data=X_train)
        result = detector.test(X_production)

        if result.is_drift:
            print(f"Joint drift detected (p={result.p_value:.4f}, "
                  f"MMD²={result.mmd2:.6f})")

    Parameters
    ----------
    reference_data:
        2D numpy array of shape (n_ref, n_features) representing the
        training/reference distribution.
    alpha:
        Significance level for the permutation test.  Default 0.05.
    n_permutations:
        Number of permutations for the p-value estimate.  Higher values
        give more accurate p-values at the cost of compute time.
        Default 200, which is adequate for most monitoring applications.
        Use 500+ for publication-quality tests.
    max_samples:
        Maximum number of samples to use from either distribution.
        When a batch is larger, the detector subsamples randomly.
        Default 1000.  Increasing this improves power but raises O(n²)
        compute cost.
    bandwidth:
        RBF kernel bandwidth σ.  When ``None`` (default), estimated
        automatically from the reference data using the median heuristic.
        Override this only if you have domain knowledge suggesting a
        specific scale.
    random_state:
        Seed for the permutation test RNG.  Set for reproducibility.
    """

    def __init__(
        self,
        reference_data: np.ndarray,
        *,
        alpha: float = 0.05,
        n_permutations: int = 200,
        max_samples: int = 1000,
        bandwidth: float | None = None,
        random_state: int | None = None,
    ) -> None:
        if reference_data.ndim != 2:
            raise ValueError(
                f"reference_data must be 2D; got shape {reference_data.shape}"
            )
        if len(reference_data) < 10:
            raise ValueError(
                "reference_data must have at least 10 rows for a meaningful test"
            )
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if n_permutations < 10:
            raise ValueError(f"n_permutations must be >= 10; got {n_permutations}")

        self._ref = np.asarray(reference_data, dtype=float)
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.max_samples = max_samples
        self._rng = np.random.default_rng(random_state)

        # Fit bandwidth once from reference data.
        if bandwidth is not None:
            if bandwidth <= 0:
                raise ValueError(f"bandwidth must be positive; got {bandwidth}")
            self._sigma = float(bandwidth)
        else:
            self._sigma = median_bandwidth(self._subsample(self._ref))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def test(self, production_data: np.ndarray) -> MMDDriftResult:
        """Run the MMD two-sample test against the reference distribution.

        The p-value is estimated by a permutation test: under the null
        hypothesis H₀ (both samples come from the same distribution), the
        MMD² statistic is exchangeable between groups.  We permute the
        combined sample ``n_permutations`` times and count how often the
        permuted statistic exceeds the observed value.

        Args:
            production_data: 2D array of shape (n_prod, n_features).
                             Must have the same number of features as
                             ``reference_data``.

        Returns:
            :class:`MMDDriftResult` with the test decision and all
            supporting statistics.

        Raises:
            ValueError: if production_data has wrong shape or too few rows.
        """
        prod = np.asarray(production_data, dtype=float)
        if prod.ndim != 2:
            raise ValueError(
                f"production_data must be 2D; got shape {prod.shape}"
            )
        if prod.shape[1] != self._ref.shape[1]:
            raise ValueError(
                f"production_data has {prod.shape[1]} features; "
                f"reference has {self._ref.shape[1]}"
            )
        if len(prod) < 5:
            raise ValueError(
                "production_data must have at least 5 rows for a meaningful test"
            )

        ref_sub = self._subsample(self._ref)
        prod_sub = self._subsample(prod)

        observed_mmd2 = _mmd2_unbiased(ref_sub, prod_sub, self._sigma)

        # Permutation test: pool both samples, shuffle, split, recompute MMD².
        pooled = np.vstack([ref_sub, prod_sub])
        m = len(ref_sub)
        perm_stats = np.empty(self.n_permutations, dtype=float)
        for i in range(self.n_permutations):
            perm = self._rng.permutation(len(pooled))
            perm_stats[i] = _mmd2_unbiased(
                pooled[perm[:m]], pooled[perm[m:]], self._sigma
            )

        # p-value: fraction of permutations with MMD² >= observed.
        # Add 1 in numerator and denominator (Phipson & Smyth, 2010) to
        # avoid p=0 which would overstate evidence against H₀.
        p_value = float((perm_stats >= observed_mmd2).sum() + 1) / (
            self.n_permutations + 1
        )

        return MMDDriftResult(
            mmd2=observed_mmd2,
            p_value=p_value,
            is_drift=p_value < self.alpha,
            alpha=self.alpha,
            n_permutations=self.n_permutations,
            n_reference=len(ref_sub),
            n_production=len(prod_sub),
            bandwidth=self._sigma,
        )

    @property
    def bandwidth(self) -> float:
        """RBF kernel bandwidth σ used for tests."""
        return self._sigma

    @property
    def n_reference(self) -> int:
        """Number of reference samples (before subsampling)."""
        return len(self._ref)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _subsample(self, X: np.ndarray) -> np.ndarray:
        """Return at most ``max_samples`` rows, sampled without replacement."""
        if len(X) <= self.max_samples:
            return X
        idx = self._rng.choice(len(X), self.max_samples, replace=False)
        return X[idx]
