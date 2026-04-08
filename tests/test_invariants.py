"""
Tests for monitoring/invariants.py.

These functions are called on every aggregation pass. A silent failure here
means the aggregation loop continues with out-of-range values, producing
nonsensical trust scores and decisions. They must raise explicitly and loudly.
"""
from __future__ import annotations

import pytest

from model_monitor.monitoring.invariants import (
    InvariantViolation,
    assert_bounded,
    assert_finite,
    assert_monotonic,
    validate_trust_components,
)

# ---------------------------------------------------------------------------
# assert_finite
# ---------------------------------------------------------------------------

def test_assert_finite_passes_on_normal_float() -> None:
    assert_finite("x", 0.5)  # must not raise


def test_assert_finite_raises_on_nan() -> None:
    with pytest.raises(InvariantViolation, match="finite"):
        assert_finite("accuracy", float("nan"))


def test_assert_finite_raises_on_positive_inf() -> None:
    with pytest.raises(InvariantViolation, match="finite"):
        assert_finite("drift", float("inf"))


def test_assert_finite_raises_on_negative_inf() -> None:
    with pytest.raises(InvariantViolation, match="finite"):
        assert_finite("latency", float("-inf"))


# ---------------------------------------------------------------------------
# assert_bounded
# ---------------------------------------------------------------------------

def test_assert_bounded_passes_at_lower_bound() -> None:
    assert_bounded("x", 0.0, lo=0.0, hi=1.0)


def test_assert_bounded_passes_at_upper_bound() -> None:
    assert_bounded("x", 1.0, lo=0.0, hi=1.0)


def test_assert_bounded_passes_in_middle() -> None:
    assert_bounded("x", 0.5, lo=0.0, hi=1.0)


def test_assert_bounded_raises_below_lower() -> None:
    with pytest.raises(InvariantViolation):
        assert_bounded("accuracy", -0.001, lo=0.0, hi=1.0)


def test_assert_bounded_raises_above_upper() -> None:
    with pytest.raises(InvariantViolation):
        assert_bounded("f1", 1.001, lo=0.0, hi=1.0)


def test_assert_bounded_raises_on_nan() -> None:
    """NaN passes the range check (nan comparisons are always False) but
    assert_bounded calls assert_finite first - so it must still raise."""
    with pytest.raises(InvariantViolation, match="finite"):
        assert_bounded("x", float("nan"), lo=0.0, hi=1.0)


# ---------------------------------------------------------------------------
# assert_monotonic
# ---------------------------------------------------------------------------

def test_assert_monotonic_passes_on_positive() -> None:
    assert_monotonic("n_batches", 42)


def test_assert_monotonic_passes_on_zero() -> None:
    assert_monotonic("n_batches", 0)


def test_assert_monotonic_raises_on_negative() -> None:
    with pytest.raises(InvariantViolation, match="non-negative"):
        assert_monotonic("n_batches", -1)


# ---------------------------------------------------------------------------
# validate_trust_components
# ---------------------------------------------------------------------------

def test_validate_trust_components_passes_on_valid() -> None:
    validate_trust_components({
        "accuracy": 0.9,
        "f1": 0.88,
        "confidence": 0.85,
        "drift": 1.0,
        "latency": 0.0,
        "behavioral": 0.5,
    })


def test_validate_trust_components_raises_on_out_of_range() -> None:
    with pytest.raises(InvariantViolation):
        validate_trust_components({"accuracy": 1.1})


def test_validate_trust_components_raises_on_nan() -> None:
    with pytest.raises(InvariantViolation):
        validate_trust_components({"f1": float("nan")})


def test_invariant_violation_is_runtime_error() -> None:
    """
    InvariantViolation must be a RuntimeError, not ValueError.
    A violated invariant is always a system bug, not a bad input from a caller.
    Callers that catch ValueError would silently swallow invariant failures.
    """
    assert issubclass(InvariantViolation, RuntimeError)
