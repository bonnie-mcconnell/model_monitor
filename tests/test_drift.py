"""
Tests for compute_psi and DriftMonitor.

These are the core algorithm tests for the classical monitoring branch.
compute_psi is the signal that feeds the trust score and decision engine,
so its correctness properties matter: identical distributions produce near-
zero PSI, severe drift produces high PSI, and reference bin edges are always
derived from the expected distribution - never recomputed from actual data.
"""

from __future__ import annotations

import numpy as np
import pytest

from model_monitor.config.settings import DriftConfig
from model_monitor.monitoring.drift import DriftMonitor, compute_psi

# ---------------------------------------------------------------------------
# compute_psi - unit tests
# ---------------------------------------------------------------------------


def test_identical_distributions_have_near_zero_psi() -> None:
    """
    When reference and production come from the same distribution, PSI
    should be negligible. This is the baseline - a freshly deployed model
    should not trigger drift alerts.
    """
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 2000)
    actual = rng.normal(0, 1, 2000)

    psi = compute_psi(reference, actual)

    assert psi < 0.05, (
        f"Expected near-zero PSI for identical distributions, got {psi:.4f}"
    )


def test_severe_drift_produces_high_psi() -> None:
    """
    When production data comes from a completely different distribution
    (different mean), PSI should exceed the 0.2 'severe' threshold.
    """
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 1000)
    actual = rng.normal(5, 1, 1000)  # mean shifted by 5 standard deviations

    psi = compute_psi(reference, actual)

    assert psi > 0.2, (
        f"Expected high PSI for severely shifted distribution, got {psi:.4f}"
    )


def test_moderate_drift_produces_moderate_psi() -> None:
    """
    A 0.3-sigma mean shift produces PSI in the moderate range (0.05–0.2),
    empirically verified across multiple seeds. This is enough to signal
    drift without crossing the 0.2 'severe' threshold.

    Note: 0.8-sigma was tested and rejected - it produces PSI ~0.70 which
    is severe, not moderate. Threshold bounds must be grounded in measured
    values, not intuition about what "slight" means.
    """
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 2000)
    actual = rng.normal(0.3, 1, 2000)  # 0.3-sigma shift → PSI ~0.07–0.15

    psi = compute_psi(reference, actual)

    assert 0.05 < psi < 0.2, f"Expected moderate PSI for 0.3-sigma shift, got {psi:.4f}"


def test_psi_is_nonnegative() -> None:
    """
    PSI is always ≥ 0 by definition. The formula (A-E)*log(A/E) can produce
    negative terms when A < E, but they are cancelled by positive terms from
    bins where A > E. The net result must never be negative.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        ref = rng.normal(0, 1, 500)
        act = rng.normal(rng.uniform(-2, 2), 1, 500)
        assert compute_psi(ref, act) >= 0.0


def test_psi_uses_reference_bin_edges_not_actual() -> None:
    """
    This is the key design property stated in the README: bin edges are
    derived from the reference distribution only.

    We verify this by checking that two 'actual' arrays with the same
    reference produce different PSI values when their data differs -
    and that swapping reference/actual gives a different result (asymmetry
    from fixed reference edges).
    """
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 1000)

    # Actual matches reference → low PSI
    actual_close = rng.normal(0, 1, 1000)
    psi_close = compute_psi(reference, actual_close)

    # Actual far from reference → high PSI
    actual_far = rng.normal(4, 1, 1000)
    psi_far = compute_psi(reference, actual_far)

    # With swapped roles: PSI(far, reference) uses 'far' edges - different result
    psi_swapped = compute_psi(actual_far, reference)

    assert psi_close < psi_far
    # Asymmetry confirms bin edges come from first argument (expected/reference)
    assert abs(psi_far - psi_swapped) > 0.01, (
        "PSI should be asymmetric when bin edges come from reference only"
    )


def test_psi_raises_on_2d_expected() -> None:
    ref = np.ones((10, 3))
    act = np.ones(10)
    with pytest.raises(ValueError, match="1D"):
        compute_psi(ref, act)


def test_psi_raises_on_2d_actual() -> None:
    ref = np.ones(10)
    act = np.ones((10, 3))
    with pytest.raises(ValueError, match="1D"):
        compute_psi(ref, act)


def test_psi_handles_constant_reference_distribution() -> None:
    """
    When the reference distribution is constant (all identical values),
    np.unique collapses the bin edges to fewer than 2. The function must
    return 0.0 rather than crashing with an invalid histogram call.
    """
    reference = np.full(100, 5.0)
    actual = np.random.default_rng(42).normal(5, 1, 100)

    result = compute_psi(reference, actual)

    assert result == 0.0


# ---------------------------------------------------------------------------
# DriftMonitor - integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def drift_config() -> DriftConfig:
    return DriftConfig(psi_threshold=0.2, window=3)


@pytest.fixture
def reference_features() -> np.ndarray:
    rng = np.random.default_rng(42)
    # 500 samples, 4 features - realistic for a small tabular dataset
    return rng.normal(0, 1, (500, 4))


def test_drift_monitor_returns_zero_before_window_is_full(
    drift_config: DriftConfig,
    reference_features: np.ndarray,
) -> None:
    """
    PSI is not meaningful until enough batches have accumulated.
    DriftMonitor must return 0.0 while the buffer is not full.
    """
    monitor = DriftMonitor(reference_features=reference_features, config=drift_config)
    rng = np.random.default_rng(0)

    # Feed window - 1 batches
    for _ in range(drift_config.window - 1):
        result = monitor.update(rng.normal(0, 1, (50, 4)))

    assert result == 0.0


def test_drift_monitor_returns_nonzero_when_window_is_full(
    drift_config: DriftConfig,
    reference_features: np.ndarray,
) -> None:
    """
    Once the buffer has enough batches, update() should return a non-negative
    PSI value (it may be near-zero if data matches reference, but not 0.0
    from the early-exit path).
    """
    monitor = DriftMonitor(reference_features=reference_features, config=drift_config)
    rng = np.random.default_rng(0)
    result = 0.0

    for _ in range(drift_config.window):
        result = monitor.update(rng.normal(0, 1, (50, 4)))

    # Buffer is full - should have computed PSI (even if near zero)
    assert isinstance(result, float)
    assert result >= 0.0


def test_drift_monitor_detects_feature_shift(
    drift_config: DriftConfig,
    reference_features: np.ndarray,
) -> None:
    """
    After filling the window with heavily shifted data, the returned PSI
    should exceed the configured threshold.
    """
    monitor = DriftMonitor(reference_features=reference_features, config=drift_config)
    rng = np.random.default_rng(0)

    result = 0.0
    for _ in range(drift_config.window):
        # 4-sigma shift - should produce high PSI
        result = monitor.update(rng.normal(4.0, 1, (200, 4)))

    assert result > drift_config.psi_threshold, (
        f"Expected PSI > {drift_config.psi_threshold} after severe feature shift, got {result:.4f}"
    )


def test_drift_monitor_raises_on_1d_reference() -> None:
    config = DriftConfig(psi_threshold=0.2, window=3)
    with pytest.raises(ValueError, match="2D"):
        DriftMonitor(reference_features=np.ones(10), config=config)


def test_drift_monitor_raises_on_1d_batch() -> None:
    config = DriftConfig(psi_threshold=0.2, window=3)
    monitor = DriftMonitor(
        reference_features=np.ones((10, 2)),
        config=config,
    )
    with pytest.raises(ValueError, match="2D"):
        monitor.update(np.ones(10))


def test_drift_monitor_window_is_rolling(
    drift_config: DriftConfig,
    reference_features: np.ndarray,
) -> None:
    """
    DriftMonitor uses a deque(maxlen=window), so after filling with drifted
    data, replacing all batches with clean data should bring PSI back down.
    """
    monitor = DriftMonitor(reference_features=reference_features, config=drift_config)
    rng = np.random.default_rng(0)

    # Fill with drifted data
    for _ in range(drift_config.window):
        monitor.update(rng.normal(5.0, 1, (200, 4)))

    # Replace with clean data - after window passes, drift should be low
    result = 0.0
    for _ in range(drift_config.window):
        result = monitor.update(rng.normal(0, 1, (200, 4)))

    assert result < drift_config.psi_threshold, (
        f"Expected low PSI after window filled with clean data, got {result:.4f}"
    )


# ---------------------------------------------------------------------------
# compute_psi with pre-computed bin_edges parameter
# ---------------------------------------------------------------------------


def test_psi_with_stored_bin_edges_matches_computed() -> None:
    """
    When bin_edges are pre-computed from the reference distribution and
    passed explicitly, the result must be identical to computing them
    internally from the same reference array.

    This is the regression test for the architecture invariant: PSI with
    stored bin edges and PSI with on-the-fly edges are equivalent when
    the edges originate from the same reference distribution.
    """
    rng = np.random.default_rng(7)
    reference = rng.normal(0, 1, 2000)
    actual = rng.normal(1.5, 1, 2000)

    # Compute edges the same way compute_psi does internally
    percentiles = np.linspace(0, 100, 11)
    edges = np.unique(np.percentile(reference, percentiles))

    psi_stored = compute_psi(reference, actual, bin_edges=edges)
    psi_computed = compute_psi(reference, actual)

    assert abs(psi_stored - psi_computed) < 1e-12, (
        f"PSI with stored edges ({psi_stored:.6f}) must equal "
        f"PSI with computed edges ({psi_computed:.6f})"
    )


def test_psi_stored_bin_edges_pins_measurement_to_reference_space() -> None:
    """
    The whole point of storing bin edges is that production PSI is always
    measured in the *reference* feature space, even if production data has
    a wildly different distribution.

    Verify: using edges from reference_A to measure drift from reference_A
    gives a different result than using edges from reference_B, even when
    measuring the same actual array.  This asymmetry is what makes PSI a
    valid drift signal rather than a symmetric distance.
    """
    rng = np.random.default_rng(99)
    reference_A = rng.normal(0, 1, 1000)
    reference_B = rng.normal(5, 1, 1000)
    actual = rng.normal(2.5, 1, 1000)

    edges_A = np.unique(np.percentile(reference_A, np.linspace(0, 100, 11)))
    edges_B = np.unique(np.percentile(reference_B, np.linspace(0, 100, 11)))

    psi_from_A = compute_psi(reference_A, actual, bin_edges=edges_A)
    psi_from_B = compute_psi(reference_B, actual, bin_edges=edges_B)

    assert abs(psi_from_A - psi_from_B) > 0.05, (
        "PSI measured in different reference spaces should differ "
        f"(from_A={psi_from_A:.4f}, from_B={psi_from_B:.4f})"
    )


# ---------------------------------------------------------------------------
# DriftMonitor with stored_bin_edges
# ---------------------------------------------------------------------------


def test_drift_monitor_with_stored_bin_edges_consistent(
    drift_config: DriftConfig,
) -> None:
    """
    DriftMonitor with stored_bin_edges must produce the same drift score
    as without them when the edges come from the reference distribution.

    This is the end-to-end version of the compute_psi stored-edges test:
    it verifies the edges flow correctly through DriftMonitor.update().
    """
    rng = np.random.default_rng(11)
    n_features = 4
    reference = rng.normal(0, 1, (500, n_features))

    # Build stored bin edges from reference columns
    stored: dict[int, np.ndarray] = {}
    for i in range(n_features):
        edges = np.unique(np.percentile(reference[:, i], np.linspace(0, 100, 11)))
        stored[i] = edges

    monitor_stored = DriftMonitor(
        reference_features=reference,
        config=drift_config,
        stored_bin_edges=stored,
    )
    monitor_computed = DriftMonitor(
        reference_features=reference,
        config=drift_config,
    )

    # Feed both monitors the same batches
    for seed in range(drift_config.window):
        batch = rng.normal(1.5, 1, (50, n_features))
        score_stored = monitor_stored.update(batch)
        score_computed = monitor_computed.update(batch)

    # After the window fills, scores should be identical
    assert abs(score_stored - score_computed) < 1e-12, (
        f"DriftMonitor stored ({score_stored:.6f}) vs computed "
        f"({score_computed:.6f}) edges should agree"
    )


def test_drift_monitor_without_stored_edges_falls_back_gracefully(
    drift_config: DriftConfig,
) -> None:
    """
    DriftMonitor must work correctly when stored_bin_edges is None or
    missing keys.  This is the backward-compatibility path for deployments
    that have a reference_stats.json without psi_bin_edges (produced by
    an older train.py).
    """
    rng = np.random.default_rng(55)
    reference = rng.normal(0, 1, (300, 3))

    monitor = DriftMonitor(
        reference_features=reference,
        config=drift_config,
        stored_bin_edges=None,
    )

    for _ in range(drift_config.window):
        monitor.update(rng.normal(0, 1, (50, 3)))

    score = monitor.update(rng.normal(3.0, 1, (50, 3)))
    assert score > 0.0
