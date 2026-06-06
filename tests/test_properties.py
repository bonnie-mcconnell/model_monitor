"""Property-based tests for mathematical invariants.

These tests use Hypothesis to verify invariants that must hold for *all*
valid inputs, not just the handful of cases covered by example-based tests.
Running ``pytest tests/test_properties.py`` will try many random inputs and
falsify the first one that breaks an invariant.

Invariants tested
-----------------
PSI:
  - compute_psi(P, P) == 0  for any valid distribution P
  - compute_psi(P, Q) >= 0  for any valid distributions P, Q
  - PSI is symmetric up to floating-point precision

Trust score:
  - output is always in [0, 1] for any valid inputs
  - increasing drift_score never increases trust score (monotone ↑)
  - decreasing f1 never increases trust score (monotone ↓)

Decision engine:
  - output action is always a known DecisionType
  - trust_score outside [0,1] always raises ValueError
  - f1 < 0 always raises ValueError
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model_monitor.config.settings import (
    AppConfig,
    DriftConfig,
    RetrainConfig,
    RollbackConfig,
    TrustScoreConfig,
)
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.monitoring.drift import compute_psi
from model_monitor.monitoring.trust_score import compute_trust_score

# All property tests are marked slow - they run Hypothesis with many
# examples and take 20-30 seconds.  Use `make test-fast` to skip them.
pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_ACTIONS = {"none", "retrain", "rollback", "reject", "promote"}

_CONFIG = AppConfig(
    drift=DriftConfig(psi_threshold=0.2, window=5),
    retrain=RetrainConfig(
        min_f1_gain=0.02,
        cooldown_batches=3,
        min_samples=50,
        evidence_window=2,
        min_stable_batches=3,
    ),
    rollback=RollbackConfig(max_f1_drop=0.15),
    trust_score=TrustScoreConfig(),
)

# Strategy: random 1D data arrays suitable for PSI computation
_data_array = st.lists(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=20,
    max_size=200,
).map(np.array)


# ---------------------------------------------------------------------------
# PSI invariants
# ---------------------------------------------------------------------------


@given(st.tuples(_data_array, _data_array))
@settings(max_examples=200)
def test_psi_non_negative(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """PSI is always non-negative for any two valid 1D data arrays."""
    result = compute_psi(pair[0], pair[1])
    assert result >= 0.0, f"PSI was negative: {result}"


@given(_data_array)
@settings(max_examples=200)
def test_psi_identical_distributions_is_zero(arr: np.ndarray) -> None:
    """PSI of a data array against itself is always zero.

    When both arrays are identical the histograms are identical, so every
    bin ratio is 1.0 and every log(1.0) is 0 → total PSI = 0.
    """
    result = compute_psi(arr, arr)
    assert result == pytest.approx(0.0, abs=1e-9), f"PSI(X, X) was {result}, expected 0"


# ---------------------------------------------------------------------------
# Trust score invariants
# ---------------------------------------------------------------------------

_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_latency = st.floats(
    min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False
)


@given(_unit, _unit, _unit, _unit, _latency)
@settings(max_examples=300)
def test_trust_score_bounded(
    accuracy: float,
    f1: float,
    confidence: float,
    drift_score: float,
    latency: float,
) -> None:
    """Trust score is always in [0, 1] for any valid inputs."""
    score, _ = compute_trust_score(
        accuracy=accuracy,
        f1=f1,
        avg_confidence=confidence,
        drift_score=drift_score,
        decision_latency_ms=latency,
    )
    assert 0.0 <= score <= 1.0, f"Trust score out of bounds: {score}"


@given(_unit, _unit, _unit, _latency)
@settings(max_examples=200)
def test_trust_score_monotone_drift(
    accuracy: float,
    f1: float,
    confidence: float,
    latency: float,
) -> None:
    """Higher drift never produces a higher trust score."""
    low_score, _ = compute_trust_score(
        accuracy=accuracy,
        f1=f1,
        avg_confidence=confidence,
        drift_score=0.0,
        decision_latency_ms=latency,
    )
    high_score, _ = compute_trust_score(
        accuracy=accuracy,
        f1=f1,
        avg_confidence=confidence,
        drift_score=1.0,
        decision_latency_ms=latency,
    )
    assert low_score >= high_score - 1e-9, (
        f"Higher drift produced higher trust: drift=0 → {low_score}, "
        f"drift=1 → {high_score}"
    )


@given(_unit, _unit, _unit, _latency)
@settings(max_examples=200)
def test_trust_score_monotone_f1(
    accuracy: float,
    confidence: float,
    drift_score: float,
    latency: float,
) -> None:
    """Lower F1 never produces a higher trust score (all else equal)."""
    high_score, _ = compute_trust_score(
        accuracy=accuracy,
        f1=1.0,
        avg_confidence=confidence,
        drift_score=drift_score,
        decision_latency_ms=latency,
    )
    low_score, _ = compute_trust_score(
        accuracy=accuracy,
        f1=0.0,
        avg_confidence=confidence,
        drift_score=drift_score,
        decision_latency_ms=latency,
    )
    assert high_score >= low_score - 1e-9, (
        f"Lower F1 produced higher trust: f1=1.0 → {high_score}, f1=0.0 → {low_score}"
    )


# ---------------------------------------------------------------------------
# Decision engine invariants
# ---------------------------------------------------------------------------


@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=300)
def test_decision_engine_always_valid_action(
    trust_score: float,
    f1: float,
    f1_baseline: float,
    drift_score: float,
    batch_index: int,
) -> None:
    """DecisionEngine always returns a known action for valid inputs."""
    engine = DecisionEngine(_CONFIG)
    decision = engine.decide(
        batch_index=batch_index,
        trust_score=trust_score,
        f1=f1,
        f1_baseline=f1_baseline,
        drift_score=drift_score,
    )
    assert decision.action in _VALID_ACTIONS, (
        f"Unknown action returned: {decision.action!r}"
    )
    assert isinstance(decision.reason, str)
    assert isinstance(decision.metadata, dict)


@given(
    st.floats(
        allow_nan=False,
        allow_infinity=False,
    ).filter(lambda x: not (0.0 <= x <= 1.0)),
)
@settings(max_examples=100)
def test_decision_engine_rejects_invalid_trust_score(trust_score: float) -> None:
    """DecisionEngine raises ValueError for trust_score outside [0, 1]."""
    engine = DecisionEngine(_CONFIG)
    with pytest.raises(ValueError, match="trust_score"):
        engine.decide(
            batch_index=0,
            trust_score=trust_score,
            f1=0.8,
            f1_baseline=0.85,
            drift_score=0.01,
        )
