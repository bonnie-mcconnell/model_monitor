"""Tests for the regression monitoring module (monitoring.regression).

Tests verify:
  - wasserstein1_distance: closed-form property checks
  - RegressionConformalMonitor: calibration and coverage guarantees
  - compute_regression_trust_score: component logic and weight contract
  - RegressionMonitor: end-to-end integration
"""

from __future__ import annotations

import numpy as np
import pytest

from model_monitor.monitoring.regression import (
    ConformalIntervalResult,
    RegressionBatchResult,
    RegressionConformalMonitor,
    RegressionMonitor,
    compute_regression_trust_score,
    wasserstein1_distance,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def linear_model_and_data() -> tuple[
    object, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Simple linear regression fixture: y = 2x + noise."""
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 4))
    y = 2.0 * X[:, 0] + rng.standard_normal(600) * 0.5

    model = LinearRegression().fit(X[:400], y[:400])
    return model, X[:400], y[:400], X[400:], y[400:]


# ---------------------------------------------------------------------------
# wasserstein1_distance
# ---------------------------------------------------------------------------


class TestWasserstein1Distance:
    def test_zero_for_identical_arrays(self) -> None:
        x = np.random.default_rng(0).standard_normal(500)
        assert wasserstein1_distance(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_known_value_shifted_gaussian(self) -> None:
        """W₁(N(0,1), N(δ,1)) ≈ δ for a mean shift of δ.

        This is an exact result for Gaussian distributions with the same
        variance: W₁ = |μ₁ − μ₂|.
        """
        rng = np.random.default_rng(7)
        delta = 2.0
        ref = rng.standard_normal(5000)
        prod = rng.standard_normal(5000) + delta
        # Allow ±5% tolerance for finite-sample estimation.
        assert wasserstein1_distance(ref, prod) == pytest.approx(delta, rel=0.05)

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.standard_normal(300)
        b = rng.standard_normal(300) * 2.0
        assert wasserstein1_distance(a, b) == pytest.approx(
            wasserstein1_distance(b, a), rel=1e-6
        )

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(9)
        assert (
            wasserstein1_distance(rng.standard_normal(200), rng.standard_normal(200))
            >= 0.0
        )

    def test_raises_on_empty_ref(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            wasserstein1_distance(np.array([]), np.array([1.0, 2.0]))

    def test_raises_on_empty_prod(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            wasserstein1_distance(np.array([1.0, 2.0]), np.array([]))

    def test_different_length_arrays_accepted(self) -> None:
        """Wasserstein supports samples of different sizes via quantile interp."""
        rng = np.random.default_rng(5)
        ref = rng.standard_normal(300)
        prod = rng.standard_normal(100)
        d = wasserstein1_distance(ref, prod)
        assert d >= 0.0


# ---------------------------------------------------------------------------
# RegressionConformalMonitor
# ---------------------------------------------------------------------------


class TestRegressionConformalMonitor:
    def test_raises_before_calibration(self) -> None:
        m = RegressionConformalMonitor(alpha=0.10)
        with pytest.raises(RuntimeError, match="calibrate"):
            m.monitor(np.ones(10), np.ones(10))

    def test_is_calibrated_false_before_calibrate(self) -> None:
        assert RegressionConformalMonitor().is_calibrated is False

    def test_is_calibrated_true_after_calibrate(self) -> None:
        rng = np.random.default_rng(0)
        m = RegressionConformalMonitor(alpha=0.10)
        y = rng.standard_normal(100)
        yhat = y + rng.standard_normal(100) * 0.3
        m.calibrate(y, yhat)
        assert m.is_calibrated

    def test_q_hat_positive(self) -> None:
        rng = np.random.default_rng(0)
        m = RegressionConformalMonitor(alpha=0.10)
        y = rng.standard_normal(100)
        yhat = y + rng.standard_normal(100) * 0.5
        m.calibrate(y, yhat)
        assert m.q_hat is not None
        assert m.q_hat > 0.0

    def test_calibrate_raises_on_too_few_samples(self) -> None:
        m = RegressionConformalMonitor()
        with pytest.raises(ValueError, match="at least 5"):
            m.calibrate(np.ones(3), np.ones(3))

    def test_calibrate_raises_on_length_mismatch(self) -> None:
        m = RegressionConformalMonitor()
        with pytest.raises(ValueError, match="same length"):
            m.calibrate(np.ones(10), np.ones(8))

    def test_coverage_at_least_nominal(self) -> None:
        """Split-conformal guarantee: coverage ≥ 1 − alpha on same distribution.

        We calibrate on 200 samples, then test on 800 samples from the same
        distribution.  The empirical coverage must be ≥ 0.90 (with high prob).
        """
        rng = np.random.default_rng(42)
        alpha = 0.10
        y_cal = rng.standard_normal(200)
        yhat_cal = y_cal + rng.standard_normal(200) * 0.4

        y_test = rng.standard_normal(800)
        yhat_test = y_test + rng.standard_normal(800) * 0.4

        m = RegressionConformalMonitor(alpha=alpha)
        m.calibrate(y_cal, yhat_cal)
        result = m.monitor(y_test, yhat_test)

        assert result.coverage_rate is not None
        assert result.coverage_rate >= (1.0 - alpha) - 0.05  # 5% tolerance

    def test_monitor_without_labels_omits_coverage(self) -> None:
        rng = np.random.default_rng(0)
        y_cal = rng.standard_normal(100)
        yhat_cal = y_cal + rng.standard_normal(100) * 0.3
        m = RegressionConformalMonitor()
        m.calibrate(y_cal, yhat_cal)
        result = m.monitor(None, rng.standard_normal(50))
        assert result.coverage_rate is None

    def test_mean_width_equals_two_times_half_width(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)
        yhat = y + rng.standard_normal(100) * 0.4
        m = RegressionConformalMonitor(alpha=0.10)
        m.calibrate(y, yhat)
        result = m.monitor(None, rng.standard_normal(50))
        assert result.mean_width == pytest.approx(2.0 * result.half_width)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            RegressionConformalMonitor(alpha=1.5)


# ---------------------------------------------------------------------------
# compute_regression_trust_score
# ---------------------------------------------------------------------------


class TestComputeRegressionTrustScore:
    def test_perfect_model_score_is_one(self) -> None:
        """Zero MAE, zero RMSE, zero Wasserstein, full coverage → score ≈ 1."""
        score, components = compute_regression_trust_score(
            mae=0.0,
            rmse=0.0,
            wasserstein=0.0,
            coverage_rate=1.0,
            data_quality_score=1.0,
        )
        assert score == pytest.approx(1.0)

    def test_all_none_defaults_to_one(self) -> None:
        """When no metrics are available all components default to 1.0."""
        score, _ = compute_regression_trust_score(
            mae=None,
            rmse=None,
            wasserstein=None,
            coverage_rate=None,
            data_quality_score=None,
        )
        assert score == pytest.approx(1.0)

    def test_score_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            score, _ = compute_regression_trust_score(
                mae=float(rng.uniform(0, 2)),
                rmse=float(rng.uniform(0, 2)),
                wasserstein=float(rng.uniform(0, 1)),
                coverage_rate=float(rng.uniform(0.5, 1.0)),
                data_quality_score=float(rng.uniform(0.5, 1.0)),
            )
            assert 0.0 <= score <= 1.0

    def test_mae_exceeds_baseline_zeroes_component(self) -> None:
        """MAE ≥ mae_baseline → mae_component = 0, trust degrades."""
        score_bad, components_bad = compute_regression_trust_score(
            mae=2.0,
            rmse=None,
            wasserstein=None,
            coverage_rate=None,
            data_quality_score=None,
            mae_baseline=1.0,
        )
        assert components_bad.mae_component == pytest.approx(0.0)
        score_good, _ = compute_regression_trust_score(
            mae=0.1,
            rmse=None,
            wasserstein=None,
            coverage_rate=None,
            data_quality_score=None,
            mae_baseline=1.0,
        )
        assert score_good > score_bad

    def test_custom_weights_respected(self) -> None:
        """Custom weights must be applied to the component values."""
        # All weights on MAE: perfect MAE → score = 1.0
        score, _ = compute_regression_trust_score(
            mae=0.0,
            rmse=5.0,  # terrible
            wasserstein=5.0,  # terrible
            coverage_rate=0.0,  # terrible
            data_quality_score=0.0,  # terrible
            mae_baseline=1.0,
            weights={
                "mae": 1.0,
                "rmse": 0.0,
                "drift": 0.0,
                "coverage": 0.0,
                "data_quality": 0.0,
            },
        )
        assert score == pytest.approx(1.0)

    def test_components_all_in_unit_interval(self) -> None:
        _, components = compute_regression_trust_score(
            mae=0.5,
            rmse=0.7,
            wasserstein=0.2,
            coverage_rate=0.88,
            data_quality_score=0.95,
        )
        for v in (
            components.mae_component,
            components.rmse_component,
            components.drift_component,
            components.coverage_component,
            components.data_quality,
        ):
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# RegressionMonitor (integration)
# ---------------------------------------------------------------------------


class TestRegressionMonitor:
    def test_predict_returns_batch_result(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, y_ref, X_test, y_test = linear_model_and_data
        ref_preds = model.predict(X_ref)
        m = RegressionMonitor(
            model.predict,
            reference_predictions=ref_preds,
            mae_baseline=0.5,
            rmse_baseline=0.6,
        )
        result = m.predict(X_test[:50], y_true=y_test[:50])
        assert isinstance(result, RegressionBatchResult)

    def test_predictions_shape(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, _, X_test, _ = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
        )
        result = m.predict(X_test[:30])
        assert result.predictions.shape == (30,)

    def test_trust_score_in_unit_interval(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, _, X_test, _ = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
        )
        result = m.predict(X_test[:50])
        assert 0.0 <= result.trust_score <= 1.0

    def test_mae_rmse_require_y_true(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, _, X_test, _ = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
        )
        result = m.predict(X_test[:30])  # no y_true
        assert result.mae is None
        assert result.rmse is None

    def test_mae_rmse_computed_with_y_true(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, _, X_test, y_test = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
        )
        result = m.predict(X_test[:50], y_true=y_test[:50])
        assert result.mae is not None and result.mae >= 0.0
        assert result.rmse is not None and result.rmse >= result.mae

    def test_wasserstein_non_negative(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, _, X_test, _ = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
        )
        result = m.predict(X_test[:50])
        assert result.wasserstein is not None
        assert result.wasserstein >= 0.0

    def test_interval_result_none_before_calibration(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, _, X_test, y_test = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
        )
        result = m.predict(X_test[:50], y_true=y_test[:50])
        assert result.interval_result is None

    def test_interval_result_present_after_calibration(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, y_ref, X_test, y_test = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
        )
        m.calibrate(y_ref[:100], model.predict(X_ref[:100]))
        result = m.predict(X_test[:50], y_true=y_test[:50])
        assert result.interval_result is not None
        assert isinstance(result.interval_result, ConformalIntervalResult)

    def test_is_healthy_property(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, _, X_test, _ = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
        )
        result = m.predict(X_test[:50])
        assert result.is_healthy == (result.trust_score >= 0.70)

    def test_n_batches_increments(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, _, X_test, _ = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
        )
        m.predict(X_test[:30])
        m.predict(X_test[30:60])
        assert m.n_batches == 2

    def test_history_newest_first(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, _, X_test, _ = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
        )
        m.predict(X_test[:30], batch_id="first")
        m.predict(X_test[30:60], batch_id="second")
        assert m.history[0]["batch_id"] == "second"
        assert m.history[1]["batch_id"] == "first"

    def test_summary_contains_required_keys(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, _, X_test, y_test = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
            mae_baseline=0.5,
        )
        m.predict(X_test[:50], y_true=y_test[:50])
        s = m.summary()
        for key in (
            "n_batches",
            "mean_trust_score",
            "mean_mae",
            "mean_rmse",
            "mean_wasserstein",
            "mae_baseline",
        ):
            assert key in s, f"missing key: {key}"

    def test_summary_empty_before_predict(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        model, X_ref, _, _, _ = linear_model_and_data
        m = RegressionMonitor(
            model.predict,
            reference_predictions=model.predict(X_ref),
        )
        assert m.summary() == {}

    def test_degraded_trust_on_badly_shifted_predictions(
        self,
        linear_model_and_data: tuple,
    ) -> None:
        """A model predicting nonsense should score lower than the true model."""
        model, X_ref, _, X_test, y_test = linear_model_and_data
        ref_preds = model.predict(X_ref)

        # Good monitor: real model
        m_good = RegressionMonitor(
            model.predict,
            reference_predictions=ref_preds,
            mae_baseline=0.5,
            rmse_baseline=0.7,
            w1_threshold=1.0,
        )
        r_good = m_good.predict(X_test[:50], y_true=y_test[:50])

        # Bad monitor: constant predictor always outputs 100 (large residuals)
        m_bad = RegressionMonitor(
            lambda X: np.full(len(X), 100.0),
            reference_predictions=ref_preds,
            mae_baseline=0.5,
            rmse_baseline=0.7,
            w1_threshold=1.0,
        )
        r_bad = m_bad.predict(X_test[:50], y_true=y_test[:50])

        assert r_good.trust_score > r_bad.trust_score
