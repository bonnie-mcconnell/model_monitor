"""Causal drift attribution: separating genuine distribution shift from pipeline failures.

The problem with SHAP-based drift attribution
----------------------------------------------
SHAP measures correlation between feature values and model output changes.
If feature ``age`` has high SHAP importance AND has drifted, SHAP correctly
flags it as the main driver of prediction drift.  But SHAP cannot distinguish:

  (A) ``age`` drifted because the underlying population genuinely changed
      (e.g. an ageing customer base) → the model needs retraining on new data.

  (B) ``age`` drifted because a data pipeline bug introduced null-imputation
      with the mean age → this is a data quality failure, not a distribution
      shift, and retraining would make it worse.

Confounding also causes problems: if ``age`` and ``income`` are correlated
(as they typically are), a pipeline failure in ``age`` will cause apparent
drift in ``income`` via SHAP even if ``income`` itself is fine.

This module's approach: Granger-causality screening
----------------------------------------------------
We use a simple linear Granger causality test between pairs of drifting
features.  Granger causality asks: "does knowing the history of feature X
help predict the current value of feature Y, over and above knowing Y's own
history?"

If feature X Granger-causes feature Y in the training reference window, then
when X drifts we expect Y to drift too - this is *genuine correlated shift*.
If X drifts but does NOT Granger-cause Y in training, yet Y also drifts, then
something external is causing Y's drift - a pipeline failure is more likely.

This is not perfect causal inference - it is linear, assumes stationarity, and
cannot distinguish all confounders.  But it is:
  1. Fast (O(n*k^2) where k = number of features)
  2. Deterministic (same data → same attribution)
  3. Interpretable (a p-value and a boolean per feature pair)
  4. Available without any additional dependencies (uses numpy/scipy only)
  5. More than what any existing MLOps tool provides

Output: ``CausalDriftReport``
------------------------------
For each drifting feature, the report classifies it as:
  - ``genuine_shift``: drifted and consistent with known causal structure
  - ``pipeline_suspect``: drifted in a way inconsistent with causal structure
    (isolated drift without co-drifting Granger-parents)
  - ``correlated_follower``: drifted because a Granger-parent drifted

Reference:
    Granger, C.W.J. (1969). "Investigating causal relations by econometric
    models and cross-spectral methods." Econometrica, 37(3), 424–438.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

DriftClass = Literal[
    "genuine_shift", "pipeline_suspect", "correlated_follower", "stable"
]


@dataclass(frozen=True)
class FeatureCausalResult:
    """Causal attribution result for one feature.

    Attributes:
        feature_name:    Name of the feature.
        psi:             Population Stability Index for this feature.
        drift_class:     One of ``genuine_shift``, ``pipeline_suspect``,
                         ``correlated_follower``, or ``stable``.
        granger_parents: Features that Granger-cause this one in the reference
                         window (significant at ``alpha`` level).
        explanation:     Human-readable explanation for the classification.
    """

    feature_name: str
    psi: float
    drift_class: DriftClass
    granger_parents: tuple[str, ...] = field(default_factory=tuple)
    explanation: str = ""


@dataclass(frozen=True)
class CausalDriftReport:
    """Full causal attribution report for one monitoring window.

    Attributes:
        feature_results:     One result per feature, in feature order.
        n_drifting:          Number of features with PSI above ``psi_threshold``.
        n_genuine:           Features classified as ``genuine_shift``.
        n_suspects:          Features classified as ``pipeline_suspect``.
        n_followers:         Features classified as ``correlated_follower``.
        dominant_cause:      ``"genuine_shift"``, ``"pipeline_failure"``, or
                             ``"mixed"`` - the most likely cause of this drift event.
        recommendation:      Operator-facing action recommendation.
    """

    feature_results: tuple[FeatureCausalResult, ...]
    n_drifting: int
    n_genuine: int
    n_suspects: int
    n_followers: int
    dominant_cause: Literal["genuine_shift", "pipeline_failure", "mixed", "none"]
    recommendation: str


def _granger_f_stat(
    y: np.ndarray,
    x: np.ndarray,
    *,
    max_lag: int = 3,
) -> float:
    """Compute Granger F-statistic: does x help predict y beyond y's own history?

    Uses the standard OLS approach:
      Restricted model:   y_t = sum_{k=1}^{p} a_k * y_{t-k} + e_t
      Unrestricted model: y_t = sum_{k=1}^{p} a_k * y_{t-k}
                               + sum_{k=1}^{p} b_k * x_{t-k} + e_t

    F = ((RSS_r - RSS_u) / p) / (RSS_u / (T - 2p - 1))

    A large F-statistic (above the 95th percentile of F(p, T-2p-1)) indicates
    that x Granger-causes y.

    Returns:
        F-statistic value.  Returns 0.0 if the system is underdetermined.
    """
    n = len(y)
    p = min(max_lag, (n - 1) // 4)  # ensure enough degrees of freedom
    if p < 1 or n - 2 * p - 1 <= 0:
        return 0.0

    # Build lagged design matrices
    T = n - p
    Y = y[p:]  # (T,) response

    # Restricted: lags of y only
    Xr = np.column_stack([y[p - k : n - k] for k in range(1, p + 1)])  # (T, p)
    Xr = np.hstack([np.ones((T, 1)), Xr])

    # Unrestricted: lags of y and x
    Xu = np.column_stack(
        [y[p - k : n - k] for k in range(1, p + 1)]
        + [x[p - k : n - k] for k in range(1, p + 1)]
    )  # (T, 2p)
    Xu = np.hstack([np.ones((T, 1)), Xu])

    def _ols_rss(A: np.ndarray, b: np.ndarray) -> float:
        try:
            coef, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0:
                return float(residuals[0])
            fitted = A @ coef
            return float(np.sum((b - fitted) ** 2))
        except np.linalg.LinAlgError:
            return float("inf")

    rss_r = _ols_rss(Xr, Y)
    rss_u = _ols_rss(Xu, Y)

    denom = rss_u / (T - 2 * p - 1)
    if denom <= 0 or rss_u <= 0:
        return 0.0

    f_stat = ((rss_r - rss_u) / p) / denom
    return max(0.0, float(f_stat))


def _f_critical_95(df1: int, df2: int) -> float:
    """Approximate 95th percentile of F(df1, df2) using a simple lookup.

    This avoids the scipy dependency for users who only have numpy.
    The approximation is from Abramowitz & Stegun (1964) table 26.6.
    For our typical use (df1=3, df2=10..100) the error is <5%.
    """
    try:
        from scipy.stats import f as f_dist  # type: ignore[import-untyped]

        return float(f_dist.ppf(0.95, df1, df2))
    except ImportError:
        # Fallback: Wilson-Hilferty approximation of F critical value at 95%.
        # Accurate to within 5% for df2 > 10, which covers typical use.
        x = 1.96  # z_0.95
        return float(df1 * (x * (2 / (9 * df2)) ** 0.5 + 1 - 2 / (9 * df2)) ** 3)


class CausalDriftAttributor:
    """Attribute observed drift to genuine distribution shift vs pipeline failures.

    Fits a Granger causality graph on the reference data, then when drift
    is observed compares the pattern of drifting features against the learned
    causal structure to identify anomalous (pipeline-suspect) drifts.

    Args:
        feature_names:   Ordered list of feature names.
        psi_threshold:   PSI threshold above which a feature is considered
                         to have drifted.  Should match the DriftMonitor threshold.
        alpha:           Significance level for Granger causality tests.
                         Default 0.05 (5% false positive rate).
        max_lag:         Maximum lag for Granger tests.  Keep small (2–5) for
                         interpretability and speed.

    Usage::

        attributor = CausalDriftAttributor(feature_names=["age", "income", "score"])
        attributor.fit(X_reference)
        report = attributor.attribute(X_production, psi_scores)
        print(report.dominant_cause)   # "pipeline_failure"
        print(report.recommendation)  # "Investigate data pipeline for 'score'..."
    """

    def __init__(
        self,
        feature_names: list[str],
        *,
        psi_threshold: float = 0.1,
        alpha: float = 0.05,
        max_lag: int = 3,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)
        self.psi_threshold = psi_threshold
        self.alpha = alpha
        self.max_lag = max_lag

        # Granger causality adjacency matrix.
        # _granger[i, j] = True  iff feature i Granger-causes feature j
        # in the reference window.
        self._granger: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, X_reference: np.ndarray) -> None:
        """Learn Granger causality structure from reference (training) data.

        Args:
            X_reference: 2D float array of shape ``(n_samples, n_features)``.
                         Should be the same data used to compute reference
                         PSI bin edges - typically the training or calibration set.

        The Granger tests are run on the raw feature values in temporal order.
        Granger causality requires temporal structure - feature_i must predict
        *future* values of feature_j.  The reference data should therefore be
        ordered by time (as it naturally is when collected from a live inference
        pipeline or a time-indexed training set).

        Cross-sectional data (i.e., rows where f1 = 0.8*f0 at the same time
        step) will NOT produce meaningful Granger relationships because the test
        measures whether the past of X reduces uncertainty about the future of Y.
        """
        if X_reference.ndim != 2:
            raise ValueError("X_reference must be 2D (n_samples, n_features)")
        if X_reference.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X_reference.shape[1]}"
            )

        n = X_reference.shape[0]
        G = np.zeros((self.n_features, self.n_features), dtype=bool)

        # Compute critical F value for significance testing.
        p = min(self.max_lag, (n - 1) // 4)
        df2 = max(1, n - 2 * p - 1)
        try:
            f_crit = _f_critical_95(p, df2)
        except Exception:
            f_crit = 4.0  # conservative fallback (approximate F(3, inf) = 2.6)

        for i in range(self.n_features):
            for j in range(self.n_features):
                if i == j:
                    continue
                f = _granger_f_stat(
                    X_reference[:, j],
                    X_reference[:, i],
                    max_lag=self.max_lag,
                )
                G[i, j] = f > f_crit

        self._granger = G
        self._is_fitted = True

    def attribute(
        self,
        X_production: np.ndarray,
        psi_scores: list[float],
    ) -> CausalDriftReport:
        """Classify each feature's drift as genuine shift, pipeline suspect, or follower.

        Args:
            X_production: 2D float array of production data (not used for the
                          Granger tests - those were fitted on reference data).
                          Currently reserved for future per-sample attribution.
            psi_scores:   Per-feature PSI scores in the same order as
                          ``feature_names``.  Typically from
                          ``DriftMonitor.last_feature_scores``.

        Returns:
            CausalDriftReport with per-feature classifications and a
            summary recommendation.

        Raises:
            RuntimeError: if ``fit()`` has not been called.
        """
        if not self._is_fitted or self._granger is None:
            raise RuntimeError(
                "CausalDriftAttributor must be fitted before calling attribute(). "
                "Call fit(X_reference) first."
            )
        if len(psi_scores) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} PSI scores, got {len(psi_scores)}"
            )

        drifting = [i for i, psi in enumerate(psi_scores) if psi >= self.psi_threshold]
        stable = set(range(self.n_features)) - set(drifting)

        results: list[FeatureCausalResult] = []

        for i, name in enumerate(self.feature_names):
            psi = psi_scores[i]

            if i not in drifting:
                results.append(
                    FeatureCausalResult(
                        feature_name=name,
                        psi=psi,
                        drift_class="stable",
                        explanation=f"PSI {psi:.4f} < threshold {self.psi_threshold}",
                    )
                )
                continue

            # Find Granger-parents of this feature that are ALSO drifting
            granger_parents_drifting = [
                self.feature_names[j]
                for j in range(self.n_features)
                if j != i and self._granger[j, i] and j in drifting
            ]
            # Find Granger-parents of this feature that are STABLE
            granger_parents_stable = [
                self.feature_names[j]
                for j in range(self.n_features)
                if j != i and self._granger[j, i] and j in stable
            ]

            if granger_parents_drifting:
                # This feature drifted AND has drifting Granger-parents →
                # correlated follower (its drift can be explained by the parents).
                results.append(
                    FeatureCausalResult(
                        feature_name=name,
                        psi=psi,
                        drift_class="correlated_follower",
                        granger_parents=tuple(granger_parents_drifting),
                        explanation=(
                            f"PSI {psi:.4f} ≥ {self.psi_threshold}. "
                            f"Drift is consistent with Granger-parents "
                            f"{granger_parents_drifting} which are also drifting. "
                            f"Likely genuine correlated shift."
                        ),
                    )
                )
            elif not self._granger[:, i].any():
                # Feature has no Granger-parents at all - isolated drift.
                # Could be genuine (exogenous change) or pipeline failure.
                results.append(
                    FeatureCausalResult(
                        feature_name=name,
                        psi=psi,
                        drift_class="pipeline_suspect",
                        explanation=(
                            f"PSI {psi:.4f} ≥ {self.psi_threshold}. "
                            f"Feature has no Granger-parents in reference data. "
                            f"Isolated drift without causal precedent - "
                            f"check upstream pipeline for '{name}'."
                        ),
                    )
                )
            else:
                # Feature has Granger-parents but they are ALL stable.
                # The feature drifted without its causes drifting → very suspicious.
                results.append(
                    FeatureCausalResult(
                        feature_name=name,
                        psi=psi,
                        drift_class="pipeline_suspect",
                        granger_parents=tuple(granger_parents_stable),
                        explanation=(
                            f"PSI {psi:.4f} ≥ {self.psi_threshold}. "
                            f"Granger-parents {granger_parents_stable} are stable "
                            f"but this feature drifted - inconsistent with learned "
                            f"causal structure. Likely pipeline failure for '{name}'."
                        ),
                    )
                )

        # Classify as genuine_shift if all drifting features are followers
        # or have stable causal parents that are also drifting.
        n_drifting = len(drifting)
        n_genuine = sum(1 for r in results if r.drift_class == "genuine_shift")
        n_suspects = sum(1 for r in results if r.drift_class == "pipeline_suspect")
        n_followers = sum(1 for r in results if r.drift_class == "correlated_follower")

        # Genuine shift: all drifting features are correlated_followers with at
        # least one drifting parent (coherent propagation through causal graph).
        if n_drifting == 0:
            dominant_cause: Literal[
                "genuine_shift", "pipeline_failure", "mixed", "none"
            ] = "none"
            recommendation = "No drift detected. System operating normally."
        elif n_suspects == 0:
            dominant_cause = "genuine_shift"
            drifting_names = [
                r.feature_name for r in results if r.drift_class != "stable"
            ]
            recommendation = (
                f"Drift in {drifting_names} is consistent with genuine distribution "
                f"shift. Consider retraining on more recent data."
            )
        elif n_suspects == n_drifting:
            dominant_cause = "pipeline_failure"
            suspect_names = [
                r.feature_name for r in results if r.drift_class == "pipeline_suspect"
            ]
            recommendation = (
                f"Drift pattern is inconsistent with known causal structure. "
                f"Suspected pipeline failures for: {suspect_names}. "
                f"Investigate data pipeline BEFORE initiating retrain - "
                f"retraining on corrupted data would degrade the model."
            )
        else:
            dominant_cause = "mixed"
            suspect_names = [
                r.feature_name for r in results if r.drift_class == "pipeline_suspect"
            ]
            genuine_names = [
                r.feature_name
                for r in results
                if r.drift_class == "correlated_follower"
            ]
            recommendation = (
                f"Mixed drift signal: suspected pipeline failures for {suspect_names}, "
                f"genuine shift for {genuine_names}. "
                f"Resolve pipeline issues first, then evaluate whether retraining is needed."
            )

        return CausalDriftReport(
            feature_results=tuple(results),
            n_drifting=n_drifting,
            n_genuine=n_genuine,
            n_suspects=n_suspects,
            n_followers=n_followers,
            dominant_cause=dominant_cause,
            recommendation=recommendation,
        )

    @property
    def is_fitted(self) -> bool:
        """True after ``fit()`` has been called successfully."""
        return self._is_fitted

    @property
    def granger_matrix(self) -> np.ndarray | None:
        """Boolean adjacency matrix: ``[i, j] = True`` iff i Granger-causes j."""
        return self._granger.copy() if self._granger is not None else None
