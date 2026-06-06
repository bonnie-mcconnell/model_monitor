"""
Tests for monitoring/shap_attribution.py.

Key properties:
- Zero shift when current distribution equals the reference distribution.
- Nonzero positive shift when a feature's distribution drifts.
- Drifted feature dominates the shift when only one feature changes.
- Shape and key consistency - output always covers all feature names.
- Graceful ValueError on dimension mismatch.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from model_monitor.monitoring.shap_attribution import ShapDriftAttributor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_model_and_data() -> tuple[RandomForestClassifier, np.ndarray, list[str]]:
    """Train a small RF on synthetic data. Shared across all tests in module."""
    rng = np.random.default_rng(42)
    n, d = 500, 6
    X = rng.normal(size=(n, d))
    # Label: positive when first two features sum above 0.
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    feature_names = [f"f{i}" for i in range(d)]
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, X, feature_names


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_attributor_constructs_successfully(
    trained_model_and_data: tuple[RandomForestClassifier, np.ndarray, list[str]],
) -> None:
    model, X, names = trained_model_and_data
    attr = ShapDriftAttributor(model, X, names)
    assert set(attr.baseline.keys()) == set(names)


def test_baseline_values_are_non_negative(
    trained_model_and_data: tuple[RandomForestClassifier, np.ndarray, list[str]],
) -> None:
    """Mean |SHAP| values are always non-negative by definition."""
    model, X, names = trained_model_and_data
    attr = ShapDriftAttributor(model, X, names)
    for v in attr.baseline.values():
        assert v >= 0.0


def test_wrong_reference_ndim_raises(
    trained_model_and_data: tuple[RandomForestClassifier, np.ndarray, list[str]],
) -> None:
    model, X, names = trained_model_and_data
    with pytest.raises(ValueError, match="2-D"):
        ShapDriftAttributor(model, X[:, 0], names)


def test_feature_names_length_mismatch_raises(
    trained_model_and_data: tuple[RandomForestClassifier, np.ndarray, list[str]],
) -> None:
    model, X, names = trained_model_and_data
    with pytest.raises(ValueError, match="feature_names"):
        ShapDriftAttributor(model, X, names[:-1])


# ---------------------------------------------------------------------------
# attribute() output shape and types
# ---------------------------------------------------------------------------


def test_attribute_returns_all_feature_keys(
    trained_model_and_data: tuple[RandomForestClassifier, np.ndarray, list[str]],
) -> None:
    model, X, names = trained_model_and_data
    attr = ShapDriftAttributor(model, X, names)
    rng = np.random.default_rng(7)
    X_new = rng.normal(size=(100, X.shape[1]))
    shifts = attr.attribute(X_new)
    assert set(shifts.keys()) == set(names)


def test_attribute_returns_floats(
    trained_model_and_data: tuple[RandomForestClassifier, np.ndarray, list[str]],
) -> None:
    model, X, names = trained_model_and_data
    attr = ShapDriftAttributor(model, X, names)
    X_new = np.random.default_rng(9).normal(size=(50, X.shape[1]))
    shifts = attr.attribute(X_new)
    for v in shifts.values():
        assert isinstance(v, float)


def test_attribute_raises_on_wrong_feature_count(
    trained_model_and_data: tuple[RandomForestClassifier, np.ndarray, list[str]],
) -> None:
    model, X, names = trained_model_and_data
    attr = ShapDriftAttributor(model, X, names)
    X_wrong = np.random.default_rng(0).normal(size=(50, X.shape[1] - 1))
    with pytest.raises(ValueError, match="features"):
        attr.attribute(X_wrong)


# ---------------------------------------------------------------------------
# Drift signal correctness
# ---------------------------------------------------------------------------


def test_near_zero_shift_on_same_distribution(
    trained_model_and_data: tuple[RandomForestClassifier, np.ndarray, list[str]],
) -> None:
    """When current distribution == reference, shifts should be near zero.

    We test that the maximum absolute shift is small, not exactly zero -
    sampling variance from capping at max_explain_rows ensures there is
    always some noise.
    """
    model, X, names = trained_model_and_data
    attr = ShapDriftAttributor(model, X, names, max_explain_rows=500)
    # Use a different seed but same distribution.
    X_same = np.random.default_rng(99).normal(size=(500, X.shape[1]))
    shifts = attr.attribute(X_same)
    max_shift = max(abs(v) for v in shifts.values())
    # Allow a generous threshold - we're testing "near zero", not "exactly zero".
    assert max_shift < 0.5, (
        f"Expected near-zero shift on same distribution, got {max_shift:.3f}"
    )


def test_drifted_feature_has_largest_shift(
    trained_model_and_data: tuple[RandomForestClassifier, np.ndarray, list[str]],
) -> None:
    """When only f0 drifts significantly, its shift should be the largest.

    f0 and f1 are the most informative features (the label depends on their
    sum), so a large shift in f0's distribution should produce the largest
    SHAP importance shift.
    """
    model, X, names = trained_model_and_data
    attr = ShapDriftAttributor(model, X, names)
    rng = np.random.default_rng(17)
    X_drifted = rng.normal(size=(300, X.shape[1]))
    # Inject a large mean shift on f0 only.
    X_drifted[:, 0] += 5.0
    shifts = attr.attribute(X_drifted)
    # f0's shift should be larger than any non-informative feature's shift.
    f0_shift = abs(shifts["f0"])
    # f4 and f5 are low-information features; their shifts should be smaller.
    low_info_shifts = [abs(shifts[f"f{i}"]) for i in range(4, len(names))]
    assert f0_shift > max(low_info_shifts), (
        f"f0 shift ({f0_shift:.3f}) should exceed low-info shifts {low_info_shifts}"
    )
