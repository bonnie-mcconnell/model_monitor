"""Regression monitoring: MAE, RMSE, Wasserstein output drift, conformal intervals.

The existing monitoring stack is built around binary/multiclass classification -
accuracy, F1, conformal prediction *sets*, and probability-distribution PSI.
None of these apply to regression.  This module provides the regression-specific
equivalents for every monitoring component:

  +--------------------------+--------------------------+------------------------+
  | Classification           | Regression (this module) | What changes           |
  +--------------------------+--------------------------+------------------------+
  | Accuracy / F1            | MAE / RMSE               | Error metric           |
  | Prediction set coverage  | Interval coverage        | Conformal guarantee    |
  | Prediction set size      | Interval width           | Efficiency metric      |
  | KL divergence on P(y)    | Wasserstein-1 distance   | Output drift test      |
  +--------------------------+--------------------------+------------------------+

Design notes
------------
- Wasserstein-1 (earth mover's distance) on 1D distributions has a closed-form
  solution via sorted CDFs: W₁ = mean |F_ref⁻¹(u) − F_prod⁻¹(u)|.  This is
  O(n log n) and parameter-free - no bandwidth selection, no binning artefacts.

- Conformal prediction intervals use the split-conformal method (Papadopoulos
  2002; Angelopoulos & Bates 2022): compute nonconformity scores on a held-out
  calibration set, then use the (1−α)-quantile as the symmetric half-width.
  This guarantees marginal coverage ≥ 1−α under exchangeability.

- The regression trust score replaces accuracy/F1 with MAE/RMSE components and
  replaces conformal coverage-of-sets with interval coverage rate.  All other
  components (drift, latency, data quality) are identical to classification.

References
----------
Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002). Inductive
  confidence machines for regression. ECML 2002.

Angelopoulos, A. N., & Bates, S. (2022). A gentle introduction to conformal
  prediction and distribution-free uncertainty quantification. arXiv:2107.07511.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Wasserstein-1 output drift
# ---------------------------------------------------------------------------


def wasserstein1_distance(ref: np.ndarray, prod: np.ndarray) -> float:
    """Compute the Wasserstein-1 (earth mover's) distance between two 1D samples.

    The closed-form solution for 1D distributions is::

        W₁(P, Q) = ∫|F_P(t) − F_Q(t)| dt
                 = mean |sort(P)[i] − sort(Q)[i]|  (after quantile-interpolation)

    We use quantile interpolation so the two samples need not have the same size.

    Args:
        ref:   1D array of reference predictions / residuals.
        prod:  1D array of production predictions / residuals.

    Returns:
        Non-negative float.  Zero means the distributions are identical.

    Raises:
        ValueError: if either array is empty or not 1D.
    """
    ref = np.asarray(ref, dtype=float).ravel()
    prod = np.asarray(prod, dtype=float).ravel()
    if len(ref) == 0 or len(prod) == 0:
        raise ValueError("ref and prod must be non-empty")

    quantiles = np.linspace(0.0, 1.0, min(len(ref), len(prod), 500))
    q_ref = np.quantile(ref, quantiles)
    q_prod = np.quantile(prod, quantiles)
    return float(np.mean(np.abs(q_ref - q_prod)))


# ---------------------------------------------------------------------------
# Conformal prediction intervals  (split-conformal)
# ---------------------------------------------------------------------------


@dataclass
class ConformalIntervalResult:
    """Result of one conformal interval monitoring call.

    Attributes:
        coverage_rate:    Fraction of samples whose true value falls inside
                          [ŷ − half_width, ŷ + half_width].  Should be
                          ≥ 1 − alpha under exchangeability.
        mean_width:       Mean interval width (2 × half_width).  Tracks
                          prediction efficiency; growing width signals
                          increasing epistemic uncertainty.
        half_width:       Calibrated half-width q̂ used for the intervals.
        coverage_ok:      True when coverage_rate ≥ target − 2σ_binomial.
        target_coverage:  1 − alpha, the nominal guarantee.
        n_samples:        Number of test samples evaluated.
    """

    coverage_rate: float | None
    mean_width: float
    half_width: float
    coverage_ok: bool | None
    target_coverage: float
    n_samples: int


class RegressionConformalMonitor:
    """Split-conformal prediction intervals for regression.

    At calibration time we compute nonconformity scores on a held-out
    calibration set::

        s_i = |y_i − ŷ_i|   (absolute residual)

    The calibrated half-width is then::

        q̂ = (1 + 1/n_cal) × quantile(s, 1 − alpha)

    This guarantees P(y_new ∈ [ŷ − q̂, ŷ + q̂]) ≥ 1 − alpha for any new
    exchangeable sample.

    Usage::

        monitor = RegressionConformalMonitor(alpha=0.10)
        monitor.calibrate(y_cal, y_hat_cal)          # on held-out set
        result = monitor.monitor(y_true, y_hat)       # on each production batch
        print(result.coverage_rate, result.mean_width)
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha
        self._q_hat: float | None = None
        self._n_cal: int = 0

    # -------------------------------------------------------------------------

    def calibrate(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
    ) -> None:
        """Fit the nonconformity quantile on a held-out calibration set.

        Args:
            y_true: True labels, shape (n_cal,).
            y_hat:  Model predictions, shape (n_cal,).

        Raises:
            ValueError: if arrays have different lengths or fewer than 5 samples.
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_hat = np.asarray(y_hat, dtype=float).ravel()
        if len(y_true) != len(y_hat):
            raise ValueError(
                f"y_true and y_hat must have the same length; "
                f"got {len(y_true)} and {len(y_hat)}"
            )
        if len(y_true) < 5:
            raise ValueError("Calibration requires at least 5 samples")

        scores = np.abs(y_true - y_hat)
        # Finite-sample correction: multiply by (1 + 1/n) so empirical coverage
        # is exactly ≥ 1 − alpha rather than ≥ 1 − alpha − 1/n.
        level = min((1.0 - self.alpha) * (1.0 + 1.0 / len(scores)), 1.0)
        self._q_hat = float(np.quantile(scores, level))
        self._n_cal = len(scores)

    # -------------------------------------------------------------------------

    def monitor(
        self,
        y_true: np.ndarray | None,
        y_hat: np.ndarray,
    ) -> ConformalIntervalResult:
        """Evaluate conformal interval quality on a production batch.

        Args:
            y_true: Ground-truth values, shape (n,).  When ``None``, only the
                    interval width is reported (coverage cannot be computed).
            y_hat:  Model predictions, shape (n,).

        Returns:
            :class:`ConformalIntervalResult`.

        Raises:
            RuntimeError: if called before :meth:`calibrate`.
        """
        if self._q_hat is None:
            raise RuntimeError("Call calibrate() before monitor()")

        y_hat = np.asarray(y_hat, dtype=float).ravel()
        half_width = self._q_hat
        mean_width = 2.0 * half_width
        target = 1.0 - self.alpha

        coverage_rate: float | None = None
        coverage_ok: bool | None = None
        if y_true is not None:
            y_true_arr = np.asarray(y_true, dtype=float).ravel()
            residuals = np.abs(y_true_arr - y_hat)
            coverage_rate = float((residuals <= half_width).mean())
            # Wilson binomial 95% lower bound - flag only genuine under-coverage.
            n = len(y_true_arr)
            se = np.sqrt(coverage_rate * (1.0 - coverage_rate) / max(n, 1))
            coverage_ok = coverage_rate >= target - 2.0 * se

        return ConformalIntervalResult(
            coverage_rate=coverage_rate,
            mean_width=mean_width,
            half_width=half_width,
            coverage_ok=coverage_ok,
            target_coverage=target,
            n_samples=len(y_hat),
        )

    @property
    def is_calibrated(self) -> bool:
        """True once :meth:`calibrate` has been called successfully."""
        return self._q_hat is not None

    @property
    def q_hat(self) -> float | None:
        """Calibrated nonconformity quantile (the symmetric half-width)."""
        return self._q_hat


# ---------------------------------------------------------------------------
# Regression trust score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionTrustComponents:
    """Explainable breakdown of a regression trust score.

    All components are in [0, 1] where 1 is ideal.

    Attributes:
        mae_component:       1 − clamp(MAE / mae_baseline, 0, 1).
                             Degrades linearly as MAE approaches mae_baseline.
        rmse_component:      1 − clamp(RMSE / rmse_baseline, 0, 1).
        drift_component:     1 − clamp(wasserstein / w1_threshold, 0, 1).
        coverage_component:  Interval coverage rate (or 1.0 if not available).
        data_quality:        From DataQualityMonitor (or 1.0 if not available).
    """

    mae_component: float
    rmse_component: float
    drift_component: float
    coverage_component: float
    data_quality: float


def compute_regression_trust_score(
    *,
    mae: float | None,
    rmse: float | None,
    wasserstein: float | None,
    coverage_rate: float | None,
    data_quality_score: float | None,
    mae_baseline: float = 1.0,
    rmse_baseline: float = 1.0,
    w1_threshold: float = 0.5,
    weights: dict[str, float] | None = None,
) -> tuple[float, RegressionTrustComponents]:
    """Compute an explainable trust score for a regression model batch.

    Args:
        mae:                Mean absolute error on this batch.
        rmse:               Root mean squared error on this batch.
        wasserstein:        Wasserstein-1 distance of predictions from reference.
        coverage_rate:      Conformal interval coverage rate (None = skip).
        data_quality_score: From DataQualityMonitor (None = skip).
        mae_baseline:       MAE at which the MAE component hits 0.  Set to the
                            training-time MAE.  Default 1.0 (override required
                            for meaningful scores).
        rmse_baseline:      RMSE at which the RMSE component hits 0.
        w1_threshold:       Wasserstein distance at which drift component hits 0.
        weights:            Optional dict overriding component weights.  Keys:
                            ``mae``, ``rmse``, ``drift``, ``coverage``,
                            ``data_quality``.  Must sum to ≤ 1.

    Returns:
        (trust_score, RegressionTrustComponents)
    """
    _weights: dict[str, float] = weights or {
        "mae": 0.25,
        "rmse": 0.20,
        "drift": 0.25,
        "coverage": 0.20,
        "data_quality": 0.10,
    }

    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    mae_c = _clamp(1.0 - (mae / mae_baseline)) if mae is not None else 1.0
    rmse_c = _clamp(1.0 - (rmse / rmse_baseline)) if rmse is not None else 1.0
    drift_c = _clamp(1.0 - (wasserstein / w1_threshold)) if wasserstein is not None else 1.0
    cov_c = coverage_rate if coverage_rate is not None else 1.0
    dq_c = data_quality_score if data_quality_score is not None else 1.0

    components = RegressionTrustComponents(
        mae_component=mae_c,
        rmse_component=rmse_c,
        drift_component=drift_c,
        coverage_component=cov_c,
        data_quality=dq_c,
    )

    score = (
        _weights.get("mae", 0.25) * mae_c
        + _weights.get("rmse", 0.20) * rmse_c
        + _weights.get("drift", 0.25) * drift_c
        + _weights.get("coverage", 0.20) * cov_c
        + _weights.get("data_quality", 0.10) * dq_c
    )

    return _clamp(score), components


# ---------------------------------------------------------------------------
# RegressionBatchResult
# ---------------------------------------------------------------------------


@dataclass
class RegressionBatchResult:
    """Result of one :class:`RegressionMonitor` predict call.

    Attributes:
        predictions:      Model predictions, shape (n,).
        trust_score:      Composite health score in [0, 1].
        mae:              Mean absolute error (requires y_true).
        rmse:             Root mean squared error (requires y_true).
        wasserstein:      Wasserstein-1 distance of predictions from reference.
        interval_result:  Conformal interval evaluation (requires y_true and
                          prior calibration).
        trust_components: Explainable breakdown of the trust score.
        batch_id:         Batch identifier.
    """

    predictions: np.ndarray
    trust_score: float
    mae: float | None
    rmse: float | None
    wasserstein: float | None
    interval_result: ConformalIntervalResult | None
    trust_components: RegressionTrustComponents
    batch_id: str

    @property
    def is_healthy(self) -> bool:
        """True when trust_score >= 0.70."""
        return self.trust_score >= 0.70

    @property
    def is_interval_coverage_ok(self) -> bool:
        """True when conformal interval coverage is within tolerance.

        Returns True (safe default) when conformal monitoring is not configured.
        """
        if self.interval_result is None:
            return True
        ok = self.interval_result.coverage_ok
        return ok if ok is not None else True


# ---------------------------------------------------------------------------
# RegressionMonitor
# ---------------------------------------------------------------------------


class RegressionMonitor:
    """Wrap any regression model with production-grade monitoring.

    Usage::

        from model_monitor.monitoring.regression import RegressionMonitor

        monitor = RegressionMonitor(
            model,
            reference_predictions=y_hat_train,
            mae_baseline=mae_train,
            rmse_baseline=rmse_train,
        )
        monitor.calibrate(y_cal, y_hat_cal)           # optional, enables intervals
        result = monitor.predict(X_batch, y_true=y_batch)

        print(f"trust={result.trust_score:.3f}  W1={result.wasserstein:.4f}")
        if not result.is_interval_coverage_ok:
            print("⚠ conformal coverage degraded - model may be miscalibrated")

    Parameters
    ----------
    model:
        Any callable with ``predict(X) -> np.ndarray``.
    reference_predictions:
        Training/reference prediction values used to establish the baseline
        output distribution for Wasserstein drift detection.
    mae_baseline:
        Training-time MAE.  Used to compute the MAE trust component.
        Defaults to 1.0 - set this to your actual training MAE.
    rmse_baseline:
        Training-time RMSE.  Defaults to 1.0.
    alpha:
        Significance level for conformal interval coverage.  Default 0.10
        (90% coverage guarantee).
    w1_threshold:
        Wasserstein-1 distance above which the drift component hits 0.
        Should be set to a meaningful scale for your target variable.
        Default 0.5.
    """

    def __init__(
        self,
        model: Callable[..., np.ndarray],
        *,
        reference_predictions: np.ndarray,
        mae_baseline: float = 1.0,
        rmse_baseline: float = 1.0,
        alpha: float = 0.10,
        w1_threshold: float = 0.5,
    ) -> None:
        self._model = model
        self._ref_preds = np.asarray(reference_predictions, dtype=float).ravel()
        self.mae_baseline = mae_baseline
        self.rmse_baseline = rmse_baseline
        self.w1_threshold = w1_threshold
        self._conformal = RegressionConformalMonitor(alpha=alpha)
        self._history: list[dict[str, object]] = []
        self._batch_count = 0

    # -------------------------------------------------------------------------

    def calibrate(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
    ) -> None:
        """Calibrate conformal prediction intervals on a held-out set.

        Args:
            y_true: True values for the calibration set.
            y_hat:  Model predictions on the calibration set.
        """
        self._conformal.calibrate(y_true, y_hat)

    # -------------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        y_true: np.ndarray | None = None,
        *,
        batch_id: str | None = None,
    ) -> RegressionBatchResult:
        """Run one batch through the model and update all monitors.

        Args:
            X:        Feature matrix, shape (n, d).
            y_true:   Ground-truth values.  Required for MAE/RMSE/coverage.
            batch_id: Optional batch identifier.

        Returns:
            :class:`RegressionBatchResult`.
        """
        import uuid

        bid = batch_id or f"reg_batch_{uuid.uuid4().hex[:8]}"
        self._batch_count += 1

        X_np = np.asarray(X)
        preds = np.asarray(self._model(X_np), dtype=float).ravel()

        # ── Wasserstein output drift ─────────────────────────────────────────
        w1: float | None = None
        try:
            w1 = wasserstein1_distance(self._ref_preds, preds)
        except Exception:
            pass

        # ── Supervised metrics ───────────────────────────────────────────────
        mae: float | None = None
        rmse: float | None = None
        if y_true is not None:
            y_np = np.asarray(y_true, dtype=float).ravel()
            if len(y_np) == len(preds):
                residuals = y_np - preds
                mae = float(np.mean(np.abs(residuals)))
                rmse = float(np.sqrt(np.mean(residuals**2)))

        # ── Conformal interval monitoring ────────────────────────────────────
        interval_result: ConformalIntervalResult | None = None
        if self._conformal.is_calibrated:
            y_for_conformal = (
                np.asarray(y_true, dtype=float).ravel() if y_true is not None else None
            )
            interval_result = self._conformal.monitor(y_for_conformal, preds)

        # ── Trust score ──────────────────────────────────────────────────────
        cov_rate = interval_result.coverage_rate if interval_result is not None else None
        trust, components = compute_regression_trust_score(
            mae=mae,
            rmse=rmse,
            wasserstein=w1,
            coverage_rate=cov_rate,
            data_quality_score=None,
            mae_baseline=self.mae_baseline,
            rmse_baseline=self.rmse_baseline,
            w1_threshold=self.w1_threshold,
        )

        self._history.append({
            "batch_id": bid,
            "n_samples": len(preds),
            "mae": mae,
            "rmse": rmse,
            "wasserstein": w1,
            "trust_score": trust,
            "coverage_rate": cov_rate,
        })

        return RegressionBatchResult(
            predictions=preds,
            trust_score=trust,
            mae=mae,
            rmse=rmse,
            wasserstein=w1,
            interval_result=interval_result,
            trust_components=components,
            batch_id=bid,
        )

    # -------------------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        """Return a plain-dict summary of the monitoring state so far."""
        if not self._history:
            return {}

        def _mean(key: str) -> float | None:
            vals = [
                float(r[key])  # type: ignore[arg-type]
                for r in self._history
                if r.get(key) is not None
            ]
            return float(np.mean(vals)) if vals else None

        return {
            "n_batches": self._batch_count,
            "mean_trust_score": _mean("trust_score"),
            "mean_mae": _mean("mae"),
            "mean_rmse": _mean("rmse"),
            "mean_wasserstein": _mean("wasserstein"),
            "mae_baseline": self.mae_baseline,
            "rmse_baseline": self.rmse_baseline,
        }

    @property
    def n_batches(self) -> int:
        """Number of batches processed so far."""
        return self._batch_count

    @property
    def history(self) -> list[dict[str, object]]:
        """Per-batch records, newest first."""
        return list(reversed(self._history))
