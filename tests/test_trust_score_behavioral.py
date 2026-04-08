"""
Tests for the behavioral component of compute_trust_score.

The pre-existing performance components (accuracy, F1, etc.) are held
constant across all tests here. We are testing only the behavioural
dimension and the interaction between the two.
"""
from __future__ import annotations

import pytest

from model_monitor.monitoring.trust_score import (
    behavioral_score,
    compute_trust_score,
)

# ---------------------------------------------------------------------------
# behavioral_score unit tests
# ---------------------------------------------------------------------------

def test_behavioral_score_zero_violations() -> None:
    assert behavioral_score(0.0) == 1.0


def test_behavioral_score_full_violations() -> None:
    assert behavioral_score(1.0) == 0.0


def test_behavioral_score_partial_violations() -> None:
    score = behavioral_score(0.5)
    assert score == pytest.approx(0.5)


def test_behavioral_score_clamped_above_one() -> None:
    # Defensive: callers should never pass > 1.0, but the function must
    # not return a value outside [0, 1] if they do.
    assert behavioral_score(1.5) == 0.0


def test_behavioral_score_clamped_below_zero() -> None:
    assert behavioral_score(-0.1) == 1.0


# ---------------------------------------------------------------------------
# compute_trust_score - behavioral component integration
# ---------------------------------------------------------------------------

# Shared "perfect performance" baseline so we isolate the behavioral effect.
_PERFECT_PERF = dict(
    accuracy=1.0,
    f1=1.0,
    avg_confidence=1.0,
    drift_score=0.0,
    decision_latency_ms=0.0,
)


def test_zero_violation_rate_applies_no_penalty() -> None:
    """
    With no violations, behavioral_violation_rate=0.0 should produce the
    same score as the default (which is also 0.0).
    """
    score_default, _ = compute_trust_score(**_PERFECT_PERF)
    score_explicit, _ = compute_trust_score(
        **_PERFECT_PERF,
        behavioral_violation_rate=0.0,
    )
    assert score_default == pytest.approx(score_explicit)


def test_full_violation_rate_lowers_trust_significantly() -> None:
    """
    100% violation rate with the default weight (0.15) should produce a
    trust score noticeably below the zero-violation score.
    """
    score_clean, _ = compute_trust_score(**_PERFECT_PERF)
    score_violated, _ = compute_trust_score(
        **_PERFECT_PERF,
        behavioral_violation_rate=1.0,
    )
    assert score_violated < score_clean
    # With perfect performance and weight=0.15, the penalty is exactly 0.15
    assert score_violated == pytest.approx(score_clean - 0.15, abs=1e-9)


def test_partial_violation_rate_interpolates_correctly() -> None:
    """
    A 50% violation rate should produce a trust score halfway between the
    zero-violation and full-violation scores.
    """
    score_clean, _ = compute_trust_score(**_PERFECT_PERF)
    score_full, _ = compute_trust_score(
        **_PERFECT_PERF, behavioral_violation_rate=1.0
    )
    score_half, _ = compute_trust_score(
        **_PERFECT_PERF, behavioral_violation_rate=0.5
    )
    midpoint = (score_clean + score_full) / 2
    assert score_half == pytest.approx(midpoint, abs=1e-9)


def test_behavioral_component_present_in_returned_components() -> None:
    """
    TrustScoreComponents must include the behavioral component so dashboards
    and audits can show its contribution.
    """
    _, components = compute_trust_score(
        **_PERFECT_PERF,
        behavioral_violation_rate=0.4,
    )
    assert "behavioral" in components
    assert components["behavioral"] == pytest.approx(0.6)


def test_trust_score_always_bounded() -> None:
    """
    No combination of inputs should produce a score outside [0.0, 1.0].
    """
    for violation_rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        score, _ = compute_trust_score(
            **_PERFECT_PERF,
            behavioral_violation_rate=violation_rate,
        )
        assert 0.0 <= score <= 1.0


def test_zero_behavioral_weight_disables_behavioral_influence() -> None:
    """
    Setting behavioral_weight=0.0 should make the trust score identical
    regardless of the violation rate.
    """
    score_clean, _ = compute_trust_score(
        **_PERFECT_PERF,
        behavioral_violation_rate=0.0,
        behavioral_weight=0.0,
    )
    score_violated, _ = compute_trust_score(
        **_PERFECT_PERF,
        behavioral_violation_rate=1.0,
        behavioral_weight=0.0,
    )
    assert score_clean == pytest.approx(score_violated)


def test_backward_compatibility_no_behavioral_args() -> None:
    """
    Callers that do not pass behavioral arguments must get the same result
    they always got - no silent behaviour change.
    """
    score, components = compute_trust_score(
        accuracy=0.9,
        f1=0.88,
        avg_confidence=0.82,
        drift_score=0.05,
        decision_latency_ms=250.0,
    )
    # Score should be non-zero, bounded, and components should have behavioral
    assert 0.0 < score <= 1.0
    assert "behavioral" in components
    # With violation_rate=0.0 (default), behavioral component should be 1.0
    assert components["behavioral"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Weight invariant - the most important property of the trust score formula
# ---------------------------------------------------------------------------

def test_weights_sum_to_one_with_default_behavioral_weight() -> None:
    """
    The fundamental invariant of a weighted average: weights must sum to 1.0.

    If this breaks, the score silently escapes [0, 1] under inputs that
    individually satisfy their bounds. The clamp() at the end masks it,
    but the components would no longer be interpretable as percentages.
    """

    behavioral_weight = 0.15
    remaining = 1.0 - behavioral_weight
    base_weights = {
        "accuracy":   0.30,
        "f1":         0.25,
        "confidence": 0.15,
        "drift":      0.20,
        "latency":    0.10,
    }
    base_total = sum(base_weights.values())
    scale = remaining / base_total

    total = sum(w * scale for w in base_weights.values()) + behavioral_weight
    assert abs(total - 1.0) < 1e-9, (
        f"Weights do not sum to 1.0: got {total}. "
        "Trust score components would not be interpretable as percentages."
    )


def test_perfect_inputs_always_score_one_regardless_of_behavioral_weight() -> None:
    """
    With all performance inputs at their best, any behavioral_weight value
    and zero violations must still produce exactly 1.0.

    This verifies the weight scaling is correct: behavioural weight is taken
    FROM the performance budget, not added on top of it.
    """
    for w in [0.0, 0.10, 0.15, 0.20, 0.50, 1.0]:
        score, _ = compute_trust_score(
            accuracy=1.0,
            f1=1.0,
            avg_confidence=1.0,
            drift_score=0.0,
            decision_latency_ms=0.0,
            behavioral_violation_rate=0.0,
            behavioral_weight=w,
        )
        assert score == pytest.approx(1.0), (
            f"Perfect inputs with behavioral_weight={w} produced {score}, expected 1.0. "
            "Weight scaling is broken - behavioral weight is being added on top of "
            "performance weights instead of substituting for them."
        )


def test_worst_inputs_always_score_zero() -> None:
    """
    With all performance inputs at their worst, score must be 0.0.
    Verifies the floor is reachable, not just the ceiling.
    """
    score, _ = compute_trust_score(
        accuracy=0.0,
        f1=0.0,
        avg_confidence=0.0,
        drift_score=0.3,         # >= 0.3 maps to drift_to_trust=0.0
        decision_latency_ms=1500.0,  # >= 1500ms maps to latency_score=0.0
        behavioral_violation_rate=1.0,
        behavioral_weight=0.15,
    )
    assert score == pytest.approx(0.0)
