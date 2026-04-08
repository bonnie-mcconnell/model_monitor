"""
Tests for utils/stats.py.

cosine_similarity is used by ToneConsistencyEvaluator on every evaluation.
moving_avg and entropy_from_labels are used in analytics and monitoring.
All three have edge cases that must be correct.
"""
from __future__ import annotations

import numpy as np
import pytest

from model_monitor.utils.stats import (
    cosine_similarity,
    entropy_from_labels,
    moving_avg,
)

# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors_returns_one() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_opposite_vectors_returns_negative_one() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_orthogonal_vectors_returns_zero() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_scale_invariant() -> None:
    """Multiplying a vector by a scalar must not change cosine similarity."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])   # 2 * a
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_zero_vector_returns_zero() -> None:
    """
    Cosine similarity is undefined for zero vectors (division by zero).
    The function must return 0.0 rather than nan or raise.
    Returning 0.0 means a zero embedding is treated as maximally dissimilar -
    a safe conservative choice for a monitoring evaluator.
    """
    a = np.array([1.0, 2.0, 3.0])
    zero = np.zeros(3)
    assert cosine_similarity(a, zero) == 0.0
    assert cosine_similarity(zero, a) == 0.0
    assert cosine_similarity(zero, zero) == 0.0


def test_cosine_result_bounded() -> None:
    """Result must always be in [-1.0, 1.0] for any non-zero input."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        a = rng.standard_normal(128)
        b = rng.standard_normal(128)
        result = cosine_similarity(a, b)
        assert -1.0 - 1e-9 <= result <= 1.0 + 1e-9


def test_cosine_known_value() -> None:
    """45-degree angle → cos(45°) = 1/√2 ≈ 0.7071."""
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 1.0])
    expected = 1.0 / np.sqrt(2)
    assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# moving_avg
# ---------------------------------------------------------------------------

def test_moving_avg_basic() -> None:
    result = moving_avg(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), window=3)
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_moving_avg_window_larger_than_array_returns_empty() -> None:
    result = moving_avg(np.array([1.0, 2.0]), window=5)
    assert len(result) == 0


def test_moving_avg_window_one_returns_input() -> None:
    x = np.array([1.0, 3.0, 5.0])
    result = moving_avg(x, window=1)
    np.testing.assert_array_almost_equal(result, x)


def test_moving_avg_raises_on_zero_window() -> None:
    with pytest.raises(ValueError, match="window must be > 0"):
        moving_avg(np.array([1.0, 2.0, 3.0]), window=0)


def test_moving_avg_raises_on_negative_window() -> None:
    with pytest.raises(ValueError, match="window must be > 0"):
        moving_avg(np.array([1.0, 2.0, 3.0]), window=-1)


# ---------------------------------------------------------------------------
# entropy_from_labels
# ---------------------------------------------------------------------------

def test_entropy_balanced_binary_is_log2() -> None:
    """50/50 split → maximum entropy for binary = ln(2) ≈ 0.693."""
    labels = np.array([0, 1, 0, 1, 0, 1])
    result = entropy_from_labels(labels)
    assert result == pytest.approx(np.log(2), abs=0.01)


def test_entropy_pure_class_is_near_zero() -> None:
    """All same label → zero uncertainty."""
    labels = np.array([1, 1, 1, 1, 1])
    result = entropy_from_labels(labels)
    assert result < 0.01


def test_entropy_empty_array_returns_zero() -> None:
    result = entropy_from_labels(np.array([]))
    assert result == 0.0


def test_entropy_is_nonnegative() -> None:
    for labels in [np.array([0]), np.array([0, 1]), np.array([0, 0, 1, 2])]:
        assert entropy_from_labels(labels) >= 0.0
