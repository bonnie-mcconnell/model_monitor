"""
Tests for compare_models (training/promotion.py).

The promotion decision has a specific boundary condition: the candidate
must improve by AT LEAST min_improvement, not strictly more than. This
is an >= comparison. Getting this boundary wrong means candidates that
just barely qualify get rejected, or candidates that don't qualify get
promoted. Either failure is silent - no crash, just the wrong decision.
"""
from __future__ import annotations

import pytest

from model_monitor.training.promotion import compare_models


def test_candidate_promoted_when_improvement_exceeds_threshold() -> None:
    result = compare_models(
        current_f1=0.80,
        candidate_f1=0.85,
        min_improvement=0.02,
    )
    assert result.promoted is True
    assert result.reason == "candidate_outperforms_current"


def test_candidate_not_promoted_when_improvement_below_threshold() -> None:
    result = compare_models(
        current_f1=0.80,
        candidate_f1=0.81,
        min_improvement=0.02,
    )
    assert result.promoted is False
    assert result.reason == "insufficient_improvement"


def test_candidate_promoted_exactly_at_threshold() -> None:
    """
    Boundary: improvement == min_improvement must promote.
    The condition is >= so the exact threshold qualifies.
    Off-by-one here would silently reject valid candidates.
    """
    result = compare_models(
        current_f1=0.80,
        candidate_f1=0.82,
        min_improvement=0.02,
    )
    assert result.promoted is True


def test_candidate_not_promoted_one_epsilon_below_threshold() -> None:
    """Complementary boundary: just below threshold must not promote."""
    result = compare_models(
        current_f1=0.80,
        candidate_f1=0.8199,
        min_improvement=0.02,
    )
    assert result.promoted is False


def test_improvement_field_is_correct() -> None:
    result = compare_models(
        current_f1=0.80,
        candidate_f1=0.85,
        min_improvement=0.02,
    )
    assert result.improvement == pytest.approx(0.05)


def test_candidate_not_promoted_when_worse_than_current() -> None:
    """Regression: a worse candidate must never be promoted."""
    result = compare_models(
        current_f1=0.85,
        candidate_f1=0.80,
        min_improvement=0.02,
    )
    assert result.promoted is False
    assert result.improvement == pytest.approx(-0.05)


def test_result_is_frozen_dataclass() -> None:
    """PromotionResult must be immutable - it is a value object."""
    result = compare_models(
        current_f1=0.80, candidate_f1=0.85, min_improvement=0.02
    )
    with pytest.raises((AttributeError, TypeError)):
        result.promoted = False  # type: ignore[misc]
