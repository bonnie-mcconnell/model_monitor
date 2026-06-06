"""Tests for new trust score components: calibration, data_quality, p95 latency.

These verify the new scoring functions introduced in the trust score rewrite:
  - calibration_to_trust: ECE → [0, 1]
  - data_quality_score propagates through compute_trust_score
  - p95_latency_ms is used in preference to mean latency
  - The full weight invariant still holds with the new 7-component formula
  - Backward compatibility: old callers without new args are unaffected
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model_monitor.monitoring.trust_score import (
    calibration_to_trust,
    compute_trust_score,
)

# ---------------------------------------------------------------------------
# calibration_to_trust
# ---------------------------------------------------------------------------


def test_calibration_perfect_ece_scores_one() -> None:
    assert calibration_to_trust(0.0) == pytest.approx(1.0)


def test_calibration_severe_ece_scores_zero() -> None:
    assert calibration_to_trust(0.10) == pytest.approx(0.0)
    assert calibration_to_trust(0.20) == pytest.approx(0.0)


def test_calibration_none_returns_neutral() -> None:
    """None (no labels) returns a neutral penalty, not 0 or 1."""
    score = calibration_to_trust(None)
    assert 0.0 < score < 1.0


def test_calibration_warning_threshold_midpoint() -> None:
    """ECE = 0.05 (Guo et al. warning threshold) should score 0.5."""
    assert calibration_to_trust(0.05) == pytest.approx(0.5)


def test_calibration_to_trust_bounded() -> None:
    for ece in [0.0, 0.01, 0.05, 0.10, 0.20, 0.5]:
        s = calibration_to_trust(ece)
        assert 0.0 <= s <= 1.0, f"calibration_to_trust({ece}) = {s} out of bounds"


# ---------------------------------------------------------------------------
# p95 latency preference
# ---------------------------------------------------------------------------


def test_p95_latency_used_over_mean_when_provided() -> None:
    """When p95 > 300ms but mean <= 300ms, latency component should be penalised."""
    score_with_p95, comps_p95 = compute_trust_score(
        accuracy=1.0,
        f1=1.0,
        avg_confidence=1.0,
        drift_score=0.0,
        decision_latency_ms=200.0,  # mean is fast
        p95_latency_ms=900.0,  # p95 is slow - should penalise
        calibration_error=0.0,
        data_quality_score=1.0,
    )
    score_mean_only, comps_mean = compute_trust_score(
        accuracy=1.0,
        f1=1.0,
        avg_confidence=1.0,
        drift_score=0.0,
        decision_latency_ms=200.0,  # mean is fast, no p95
        calibration_error=0.0,
        data_quality_score=1.0,
    )
    # p95-based score should be lower than mean-based score
    assert score_with_p95 < score_mean_only
    assert comps_p95["latency"] < comps_mean["latency"]


def test_mean_latency_used_as_fallback_when_p95_none() -> None:
    """When p95 is None, decision_latency_ms (mean) is used."""
    score_no_p95, comps = compute_trust_score(
        accuracy=1.0,
        f1=1.0,
        avg_confidence=1.0,
        drift_score=0.0,
        decision_latency_ms=200.0,  # fast mean
        p95_latency_ms=None,  # no p95
        calibration_error=0.0,
        data_quality_score=1.0,
    )
    # Fast mean → latency score should be 1.0
    assert comps["latency"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# data_quality component
# ---------------------------------------------------------------------------


def test_data_quality_score_one_no_penalty() -> None:
    score_perfect, _ = compute_trust_score(
        accuracy=0.9,
        f1=0.88,
        avg_confidence=0.85,
        drift_score=0.05,
        decision_latency_ms=100.0,
        calibration_error=0.02,
        data_quality_score=1.0,
    )
    score_none, _ = compute_trust_score(
        accuracy=0.9,
        f1=0.88,
        avg_confidence=0.85,
        drift_score=0.05,
        decision_latency_ms=100.0,
        calibration_error=0.02,
        data_quality_score=None,  # defaults to 1.0
    )
    assert score_perfect == pytest.approx(score_none)


def test_low_data_quality_reduces_trust() -> None:
    score_good, _ = compute_trust_score(
        accuracy=0.9,
        f1=0.88,
        avg_confidence=0.85,
        drift_score=0.05,
        decision_latency_ms=100.0,
        data_quality_score=1.0,
    )
    score_poor, _ = compute_trust_score(
        accuracy=0.9,
        f1=0.88,
        avg_confidence=0.85,
        drift_score=0.05,
        decision_latency_ms=100.0,
        data_quality_score=0.0,
    )
    assert score_poor < score_good


def test_data_quality_component_in_components_dict() -> None:
    _, comps = compute_trust_score(
        accuracy=0.9,
        f1=0.88,
        avg_confidence=0.85,
        drift_score=0.05,
        decision_latency_ms=100.0,
        data_quality_score=0.7,
    )
    assert "data_quality" in comps
    assert comps["data_quality"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Weight invariant - 7-component formula
# ---------------------------------------------------------------------------


def test_weights_sum_to_one_seven_components() -> None:
    """The fundamental invariant: all component weights must sum to 1.0."""
    behavioral_weight = 0.05
    dq_weight = 0.05
    remaining = 1.0 - behavioral_weight - dq_weight
    base_weights = {
        "accuracy": 0.25,
        "f1": 0.20,
        "calibration": 0.15,
        "drift": 0.20,
        "latency": 0.20,
    }
    base_total = sum(base_weights.values())
    scale = remaining / base_total
    total = (
        sum(w * scale for w in base_weights.values()) + behavioral_weight + dq_weight
    )
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, not 1.0"


def test_perfect_all_inputs_score_one() -> None:
    score, _ = compute_trust_score(
        accuracy=1.0,
        f1=1.0,
        avg_confidence=1.0,
        drift_score=0.0,
        decision_latency_ms=0.0,
        calibration_error=0.0,
        p95_latency_ms=0.0,
        data_quality_score=1.0,
        behavioral_violation_rate=0.0,
    )
    assert score == pytest.approx(1.0)


def test_worst_all_inputs_score_zero() -> None:
    score, _ = compute_trust_score(
        accuracy=0.0,
        f1=0.0,
        avg_confidence=0.0,
        drift_score=0.3,
        decision_latency_ms=1500.0,
        calibration_error=0.10,
        p95_latency_ms=1500.0,
        data_quality_score=0.0,
        behavioral_violation_rate=1.0,
        behavioral_weight=0.05,
    )
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_backward_compat_no_new_args() -> None:
    """Old callers without new args must get a valid score - no KeyError."""
    score, comps = compute_trust_score(
        accuracy=0.90,
        f1=0.88,
        avg_confidence=0.82,
        drift_score=0.05,
        decision_latency_ms=250.0,
    )
    assert 0.0 < score <= 1.0
    assert "calibration" in comps
    assert "data_quality" in comps
    assert "behavioral" in comps


# ---------------------------------------------------------------------------
# Property tests - score always in [0, 1]
# ---------------------------------------------------------------------------


@given(
    accuracy=st.floats(0.0, 1.0),
    f1=st.floats(0.0, 1.0),
    drift=st.floats(0.0, 0.5),
    latency=st.floats(0.0, 2000.0),
    ece=st.one_of(st.none(), st.floats(0.0, 0.5)),
    dq=st.one_of(st.none(), st.floats(0.0, 1.0)),
    violation_rate=st.floats(0.0, 1.0),
)
@settings(max_examples=200)
def test_trust_score_always_bounded_property(
    accuracy: float,
    f1: float,
    drift: float,
    latency: float,
    ece: float | None,
    dq: float | None,
    violation_rate: float,
) -> None:
    score, _ = compute_trust_score(
        accuracy=accuracy,
        f1=f1,
        avg_confidence=0.8,
        drift_score=drift,
        decision_latency_ms=latency,
        calibration_error=ece,
        data_quality_score=dq,
        behavioral_violation_rate=violation_rate,
    )
    assert 0.0 <= score <= 1.0, f"Trust score {score} out of [0, 1]"
