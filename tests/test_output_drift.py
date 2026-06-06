"""Tests for output (prediction distribution) drift monitoring.

OutputDriftMonitor uses PSI on predicted probability vectors rather than
input features.  These tests verify:
  - Buffer behaviour before and after the window fills
  - PSI increases when output distribution shifts
  - Per-class scores are stored correctly
  - is_drifting threshold property works correctly
  - Edge cases: single class, wrong input shape
"""

from __future__ import annotations

import numpy as np
import pytest

from model_monitor.monitoring.output_drift import OutputDriftMonitor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probs(n: int, n_classes: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(n_classes), size=n)
    return raw.astype(float)


def _stable_probs(n: int, n_classes: int = 2) -> np.ndarray:
    """Return a stable low-entropy distribution (class 0 dominant)."""
    rng = np.random.default_rng(42)
    p = rng.dirichlet([10.0] + [1.0] * (n_classes - 1), size=n)
    return p.astype(float)


def _shifted_probs(n: int, n_classes: int = 2) -> np.ndarray:
    """Return a shifted distribution (class 1 dominant - reversed)."""
    rng = np.random.default_rng(99)
    p = rng.dirichlet([1.0] * (n_classes - 1) + [10.0], size=n)
    return p.astype(float)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_output_drift_monitor_construction() -> None:
    ref = _make_probs(200, 2)
    monitor = OutputDriftMonitor(ref, window=3, threshold=0.1)
    assert monitor.n_classes == 2
    assert monitor.window == 3
    assert monitor.threshold == 0.1


def test_output_drift_monitor_rejects_1d_reference() -> None:
    with pytest.raises(ValueError, match="2D"):
        OutputDriftMonitor(np.array([0.5, 0.5]), window=3)


def test_output_drift_monitor_rejects_empty_reference() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        OutputDriftMonitor(np.empty((0, 2)), window=3)


# ---------------------------------------------------------------------------
# Buffer fill behaviour
# ---------------------------------------------------------------------------


def test_returns_zero_before_window_fills() -> None:
    ref = _stable_probs(500)
    monitor = OutputDriftMonitor(ref, window=3)
    for _ in range(2):
        score = monitor.update(_stable_probs(100))
        assert score == 0.0
    assert monitor.last_class_scores == []


def test_returns_nonzero_after_window_fills() -> None:
    ref = _stable_probs(500)
    monitor = OutputDriftMonitor(ref, window=3)
    for _ in range(3):
        monitor.update(_shifted_probs(100))
    assert monitor.last_class_scores != []


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def test_stable_distribution_produces_low_psi() -> None:
    """Same distribution in and out should give PSI near zero."""
    ref = _stable_probs(1000)
    monitor = OutputDriftMonitor(ref, window=3)
    for _ in range(3):
        monitor.update(_stable_probs(200))
    score = monitor.update(_stable_probs(200))
    assert score < 0.05, (
        f"Expected near-zero PSI for stable distribution, got {score:.4f}"
    )


def test_shifted_distribution_produces_high_psi() -> None:
    """Distribution shift should produce PSI > threshold."""
    ref = _stable_probs(1000)
    monitor = OutputDriftMonitor(ref, window=3, threshold=0.1)
    for _ in range(3):
        monitor.update(_shifted_probs(200))
    score = monitor.update(_shifted_probs(200))
    assert score > 0.1, f"Expected high PSI for shifted distribution, got {score:.4f}"


def test_per_class_scores_have_correct_length() -> None:
    ref = _make_probs(500, n_classes=3)
    monitor = OutputDriftMonitor(ref, window=2, threshold=0.1)
    batch = _make_probs(100, n_classes=3, seed=5)
    for _ in range(2):
        monitor.update(batch)
    assert len(monitor.last_class_scores) == 3


def test_mean_score_equals_mean_of_class_scores() -> None:
    """Return value must equal np.mean(last_class_scores)."""
    ref = _stable_probs(500)
    monitor = OutputDriftMonitor(ref, window=2)
    for _ in range(2):
        score = monitor.update(_shifted_probs(200))
    expected = float(np.mean(monitor.last_class_scores))
    assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# is_drifting property
# ---------------------------------------------------------------------------


def test_is_drifting_false_before_window_fills() -> None:
    ref = _stable_probs(500)
    monitor = OutputDriftMonitor(ref, window=5)
    monitor.update(_shifted_probs(100))
    assert not monitor.is_drifting


def test_is_drifting_true_on_large_shift() -> None:
    ref = _stable_probs(1000)
    monitor = OutputDriftMonitor(ref, window=2, threshold=0.05)
    for _ in range(2):
        monitor.update(_shifted_probs(300))
    assert monitor.is_drifting


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_1d_batch() -> None:
    ref = _make_probs(200, 2)
    monitor = OutputDriftMonitor(ref, window=2)
    with pytest.raises(ValueError, match="2D"):
        monitor.update(np.array([0.4, 0.6]))


def test_rejects_wrong_class_count() -> None:
    ref = _make_probs(200, 2)
    monitor = OutputDriftMonitor(ref, window=2)
    with pytest.raises(ValueError, match="expected 2"):
        monitor.update(_make_probs(50, 3))
