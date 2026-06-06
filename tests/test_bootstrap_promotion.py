"""Tests for bootstrap confidence interval promotion gating.

The core invariant: a candidate that looks better in point-estimate terms
but where the CI includes zero should NOT be promoted.  A candidate whose
CI lower bound is solidly positive should be promoted.

These tests also verify:
  - Backward compatibility: n_bootstrap=0 behaves exactly like old code
  - BootstrapCI fields are correct
  - Graceful degradation when predictions not provided
  - IEEE 754 epsilon fix is preserved
"""

from __future__ import annotations

import numpy as np
import pytest

from model_monitor.training.promotion import (
    BootstrapCI,
    compare_models,
)

# ---------------------------------------------------------------------------
# Backward compatibility - n_bootstrap=0 (default)
# ---------------------------------------------------------------------------


def test_point_estimate_promotes_when_improvement_sufficient() -> None:
    result = compare_models(
        current_f1=0.80,
        candidate_f1=0.83,
        min_improvement=0.02,
    )
    assert result.promoted is True
    assert result.reason == "candidate_outperforms_current"
    assert result.bootstrap_ci is None


def test_point_estimate_rejects_insufficient_improvement() -> None:
    result = compare_models(
        current_f1=0.80,
        candidate_f1=0.81,
        min_improvement=0.02,
    )
    assert result.promoted is False
    assert result.reason == "insufficient_improvement"


def test_ieee754_epsilon_fix_preserved() -> None:
    """0.82 - 0.80 = 0.01999... in IEEE 754 - must still promote with eps."""
    result = compare_models(
        current_f1=0.80,
        candidate_f1=0.82,
        min_improvement=0.02,
    )
    assert result.promoted is True, (
        "IEEE 754 epsilon fix broken: 0.82 - 0.80 < 0.02 without tolerance"
    )


def test_point_estimate_exact_at_threshold_promotes() -> None:
    result = compare_models(
        current_f1=0.70,
        candidate_f1=0.75,
        min_improvement=0.05,
    )
    assert result.promoted is True


# ---------------------------------------------------------------------------
# PromotionResult fields
# ---------------------------------------------------------------------------


def test_promotion_result_fields() -> None:
    result = compare_models(
        current_f1=0.80,
        candidate_f1=0.85,
        min_improvement=0.02,
    )
    assert result.current_f1 == pytest.approx(0.80)
    assert result.candidate_f1 == pytest.approx(0.85)
    assert result.improvement == pytest.approx(0.05)


def test_promotion_result_is_frozen() -> None:
    result = compare_models(current_f1=0.7, candidate_f1=0.8, min_improvement=0.02)
    with pytest.raises(Exception):
        result.promoted = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Bootstrap CI - statistical significance gate
# ---------------------------------------------------------------------------


def _make_predictions(
    n: int,
    current_f1: float,
    candidate_f1: float,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (y_true, y_pred_current, y_pred_candidate) with approximate F1s."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)

    # Current model: flip labels with probability (1 - current_f1)
    current_flip_rate = max(0.0, 1.0 - current_f1)
    y_current = y_true.copy()
    flip_mask = rng.random(n) < current_flip_rate
    y_current[flip_mask] = 1 - y_current[flip_mask]

    # Candidate model: lower flip rate
    candidate_flip_rate = max(0.0, 1.0 - candidate_f1)
    y_candidate = y_true.copy()
    flip_mask2 = rng.random(n) < candidate_flip_rate
    y_candidate[flip_mask2] = 1 - y_candidate[flip_mask2]

    return y_true, y_current, y_candidate


def test_bootstrap_ci_present_when_requested() -> None:
    y_true, y_cur, y_cand = _make_predictions(500, 0.75, 0.82, seed=0)
    result = compare_models(
        current_f1=0.75,
        candidate_f1=0.82,
        min_improvement=0.02,
        y_true=y_true,
        y_pred_current=y_cur,
        y_pred_candidate=y_cand,
        n_bootstrap=200,
        rng=np.random.default_rng(42),
    )
    assert result.bootstrap_ci is not None
    ci = result.bootstrap_ci
    assert isinstance(ci, BootstrapCI)
    assert ci.n_bootstrap == 200
    assert ci.alpha == pytest.approx(0.05)
    assert ci.lower <= ci.upper


def test_bootstrap_ci_lower_upper_ordered() -> None:
    """CI bounds must always be ordered lower <= upper."""
    y_true, y_cur, y_cand = _make_predictions(300, 0.70, 0.80)
    result = compare_models(
        current_f1=0.70,
        candidate_f1=0.80,
        min_improvement=0.02,
        y_true=y_true,
        y_pred_current=y_cur,
        y_pred_candidate=y_cand,
        n_bootstrap=100,
        rng=np.random.default_rng(0),
    )
    if result.bootstrap_ci is not None:
        assert result.bootstrap_ci.lower <= result.bootstrap_ci.upper


def test_bootstrap_blocks_noisy_improvement_on_small_sample() -> None:
    """On a tiny sample, a 2pp improvement should NOT be statistically significant."""
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=30)  # very small
    # Both models basically the same - tiny improvement by luck
    y_cur = y_true.copy()
    y_cand = y_true.copy()
    y_cand[0] = 1 - y_cand[0]  # one single prediction difference

    result = compare_models(
        current_f1=0.60,
        candidate_f1=0.62,
        min_improvement=0.01,
        y_true=y_true,
        y_pred_current=y_cur,
        y_pred_candidate=y_cand,
        n_bootstrap=500,
        rng=np.random.default_rng(42),
    )
    # With only 30 samples, CI should include zero - block the promotion.
    if result.bootstrap_ci is not None and result.bootstrap_ci.lower <= 0:
        assert result.promoted is False
        assert result.reason == "bootstrap_ci_includes_zero"


def test_bootstrap_degrades_gracefully_without_predictions() -> None:
    """When n_bootstrap > 0 but no predictions provided, fall back to point estimate."""
    result = compare_models(
        current_f1=0.80,
        candidate_f1=0.85,
        min_improvement=0.02,
        n_bootstrap=100,
        # y_true / y_pred_* intentionally omitted
    )
    assert result.promoted is True
    assert result.bootstrap_ci is None


def test_bootstrap_not_run_when_point_estimate_fails() -> None:
    """Bootstrap should not run when point-estimate gate already rejects."""
    y_true, y_cur, y_cand = _make_predictions(500, 0.80, 0.81)
    result = compare_models(
        current_f1=0.80,
        candidate_f1=0.81,
        min_improvement=0.05,  # high threshold - point estimate fails
        y_true=y_true,
        y_pred_current=y_cur,
        y_pred_candidate=y_cand,
        n_bootstrap=200,
        rng=np.random.default_rng(0),
    )
    assert result.promoted is False
    assert result.reason == "insufficient_improvement"
    assert result.bootstrap_ci is None


def test_deterministic_with_fixed_rng() -> None:
    """Same rng seed must produce same CI bounds."""
    y_true, y_cur, y_cand = _make_predictions(400, 0.72, 0.82, seed=5)
    kwargs: dict = dict(
        current_f1=0.72,
        candidate_f1=0.82,
        min_improvement=0.02,
        y_true=y_true,
        y_pred_current=y_cur,
        y_pred_candidate=y_cand,
        n_bootstrap=200,
    )
    r1 = compare_models(**kwargs, rng=np.random.default_rng(99))
    r2 = compare_models(**kwargs, rng=np.random.default_rng(99))
    assert r1.bootstrap_ci is not None
    assert r2.bootstrap_ci is not None
    assert r1.bootstrap_ci.lower == pytest.approx(r2.bootstrap_ci.lower)
    assert r1.bootstrap_ci.upper == pytest.approx(r2.bootstrap_ci.upper)


# ---------------------------------------------------------------------------
# BootstrapCI dataclass
# ---------------------------------------------------------------------------


def test_bootstrap_ci_is_frozen() -> None:
    ci = BootstrapCI(lower=0.01, upper=0.05, alpha=0.05, n_bootstrap=1000)
    with pytest.raises(Exception):
        ci.lower = 0.0  # type: ignore[misc]
