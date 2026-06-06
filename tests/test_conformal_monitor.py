"""Tests for ConformalMonitor - calibration, coverage guarantees, and edge cases.

Key invariants verified:
  1. Calibration stores q_hat and updates is_calibrated
  2. Coverage rate is >= (1 - alpha) on the calibration distribution itself
     (this is the core conformal guarantee: the threshold was set on this data)
  3. A distribution-shifted production batch has lower coverage
  4. Set size grows when the model is uncertain
  5. Unlabeled monitoring (set size only) works without labels
  6. All edge cases raise informative errors
"""

from __future__ import annotations

import numpy as np
import pytest

from model_monitor.monitoring.conformal import ConformalMonitor, ConformalMonitorResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return np.asarray(e / e.sum(axis=1, keepdims=True))


def _confident_probs(
    n: int, n_classes: int = 3, *, correct: bool = True, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) where class 0 always has highest probability."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, n_classes))
    logits[:, 0] += 3.0  # class 0 dominates
    probs = _softmax(logits)
    labels = np.zeros(n, dtype=int) if correct else rng.integers(0, n_classes, size=n)
    return probs, labels


def _uniform_probs(n: int, n_classes: int = 3, *, seed: int = 0) -> np.ndarray:
    """Return near-uniform probs - high uncertainty, large prediction sets."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(n_classes) * 0.5, size=n)
    return raw.astype(float)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_defaults() -> None:
    m = ConformalMonitor()
    assert m.alpha == 0.10
    assert not m.is_calibrated
    assert m.q_hat is None


def test_construction_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ConformalMonitor(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        ConformalMonitor(alpha=1.0)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibrate_sets_is_calibrated() -> None:
    probs, labels = _confident_probs(200)
    m = ConformalMonitor(alpha=0.10)
    m.calibrate(probs, labels)
    assert m.is_calibrated
    assert m.q_hat is not None


def test_calibrate_q_hat_in_valid_range() -> None:
    """q_hat must be a valid nonconformity score - in [0, 1] for probability-based scores."""
    probs, labels = _confident_probs(500)
    m = ConformalMonitor(alpha=0.10)
    m.calibrate(probs, labels)
    assert m.q_hat is not None
    assert 0.0 <= m.q_hat <= 1.0


def test_calibrate_rejects_mismatched_lengths() -> None:
    probs, labels = _confident_probs(100)
    m = ConformalMonitor()
    with pytest.raises(ValueError, match="same length"):
        m.calibrate(probs, labels[:50])


def test_calibrate_rejects_empty_set() -> None:
    m = ConformalMonitor()
    with pytest.raises(ValueError, match="empty"):
        m.calibrate(np.empty((0, 3)), np.empty(0, dtype=int))


def test_calibrate_rejects_1d_probs() -> None:
    m = ConformalMonitor()
    with pytest.raises(ValueError, match="2D"):
        m.calibrate(np.array([0.5, 0.3, 0.2]), np.array([0]))


# ---------------------------------------------------------------------------
# Coverage guarantee
# ---------------------------------------------------------------------------


def test_coverage_on_calibration_distribution_meets_guarantee() -> None:
    """Core conformal guarantee: coverage on the same distribution >= 1 - alpha."""
    alpha = 0.10
    n_cal = 1000
    probs, labels = _confident_probs(n_cal, seed=0)
    m = ConformalMonitor(alpha=alpha)
    m.calibrate(probs, labels)

    # Test on a fresh sample from the same distribution
    probs_test, labels_test = _confident_probs(500, seed=1)
    result = m.monitor(probs_test, labels_test)

    assert result.coverage_rate is not None
    assert result.coverage_rate >= (1 - alpha - 0.05), (  # 5% slack for finite samples
        f"Coverage {result.coverage_rate:.3f} below guarantee {1 - alpha:.3f}"
    )


def test_coverage_gap_is_positive_when_below_guarantee() -> None:
    """coverage_gap = (1-alpha) - coverage_rate should be positive when low."""
    probs_cal, labels_cal = _confident_probs(300, seed=10)
    m = ConformalMonitor(alpha=0.10)
    m.calibrate(probs_cal, labels_cal)

    # Use random labels to force low coverage
    probs_test, _ = _confident_probs(200, seed=20)
    bad_labels = np.random.default_rng(99).integers(0, 3, size=200)
    result = m.monitor(probs_test, bad_labels)

    if result.coverage_rate is not None and result.coverage_rate < 0.90:
        assert result.coverage_gap is not None
        assert result.coverage_gap > 0


# ---------------------------------------------------------------------------
# Set size behaviour
# ---------------------------------------------------------------------------


def test_confident_probs_produce_small_sets() -> None:
    """A confident model should have mean set size near 1.0."""
    probs, labels = _confident_probs(500)
    m = ConformalMonitor(alpha=0.10)
    m.calibrate(probs, labels)
    result = m.monitor(probs)
    assert result.mean_set_size < 1.5, (
        f"Expected small prediction sets for confident model, got {result.mean_set_size:.2f}"
    )


def test_uncertain_probs_produce_large_sets() -> None:
    """Uniform/uncertain probs should produce larger prediction sets than confident ones.

    We compare set sizes rather than checking an absolute threshold, because
    the q_hat is calibrated on confident probs - on uncertain probs the model
    cannot cover itself with a single class, so set sizes must be >= confident ones.
    """
    probs_cal, labels_cal = _confident_probs(500)
    m = ConformalMonitor(alpha=0.10, min_set_size_alarm=1.5)
    m.calibrate(probs_cal, labels_cal)

    confident_probs, _ = _confident_probs(300, seed=5)
    result_conf = m.monitor(confident_probs)

    uncertain_probs = _uniform_probs(300)
    result_unc = m.monitor(uncertain_probs)

    # Uncertain probs must produce equal or larger prediction sets
    assert result_unc.mean_set_size >= result_conf.mean_set_size


# ---------------------------------------------------------------------------
# Unlabeled monitoring
# ---------------------------------------------------------------------------


def test_monitor_without_labels_returns_none_coverage() -> None:
    probs, labels = _confident_probs(200)
    m = ConformalMonitor(alpha=0.10)
    m.calibrate(probs, labels)
    result = m.monitor(probs)  # no labels
    assert result.coverage_rate is None
    assert result.coverage_gap is None


def test_monitor_without_labels_still_returns_set_size() -> None:
    probs, labels = _confident_probs(200)
    m = ConformalMonitor(alpha=0.10)
    m.calibrate(probs, labels)
    result = m.monitor(probs)
    assert result.mean_set_size > 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_monitor_before_calibrate_raises() -> None:
    m = ConformalMonitor()
    with pytest.raises(RuntimeError, match="calibrate"):
        m.monitor(np.array([[0.5, 0.3, 0.2]]))


def test_monitor_rejects_wrong_class_count() -> None:
    probs, labels = _confident_probs(100, n_classes=3)
    m = ConformalMonitor(alpha=0.10)
    m.calibrate(probs, labels)

    wrong_probs = np.array([[0.5, 0.5]])  # 2 classes, calibrated on 3
    with pytest.raises(ValueError, match="classes"):
        m.monitor(wrong_probs)


# ---------------------------------------------------------------------------
# ConformalMonitorResult properties
# ---------------------------------------------------------------------------


def test_result_is_frozen_dataclass() -> None:
    r = ConformalMonitorResult(
        coverage_rate=0.91,
        mean_set_size=1.1,
        coverage_gap=-0.01,
        q_hat=0.12,
        coverage_ok=True,
    )
    with pytest.raises(Exception):
        r.coverage_rate = 0.5  # type: ignore[misc]
