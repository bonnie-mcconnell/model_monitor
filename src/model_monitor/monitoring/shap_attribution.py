"""Per-feature SHAP-based drift attribution.

PSI tells you *that* a feature's distribution has shifted.  SHAP attribution
answers a harder question: *which features are driving prediction change*, and
by how much?  A feature can drift heavily in an irrelevant part of its
distribution with little impact on predictions; another feature may drift only
slightly in a region the model weights heavily.  SHAP-based attribution
separates signal from noise.

Design
------
``ShapDriftAttributor`` wraps ``shap.TreeExplainer`` (fast for tree ensembles,
sub-50ms for a 200-tree RandomForest on batches up to 1000 rows) and computes
the shift in mean absolute SHAP value per feature relative to a reference
baseline computed at training time.

A positive shift on feature ``f3`` means the model is now *using* that feature
more than it did at training time - either because its distribution has changed
in a high-weight region, or because the model's decision boundary is interacting
differently with the current data.  Either way it's a signal worth surfacing.

The attributor is optional and injected.  When absent, ``Predictor`` produces
``shap_attribution`` as ``None`` in the ``MetricRecord``.  This keeps the
fast path free of any SHAP overhead.

Latency
-------
For a 200-estimator RandomForest and a 200-row batch, ``TreeExplainer`` takes
approximately 15–40ms on modern hardware.  The ``ShapDriftAttributor`` is
designed for use in batch (post-prediction) pipelines where this overhead is
acceptable.  Do not use it in real-time per-request serving paths.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ShapDriftAttributor:
    """Computes per-feature SHAP importance shift relative to a training baseline.

    Args:
        model:            Fitted sklearn tree ensemble (RandomForestClassifier,
                          GradientBoostingClassifier, etc.).  Must support
                          ``shap.TreeExplainer``.
        reference_X:      Reference feature matrix from the training set,
                          shape ``(n_samples, n_features)``.  Used to compute
                          the baseline mean ``|SHAP|`` per feature.
        feature_names:    Column names for the feature matrix.  Must have
                          ``n_features`` elements.
        max_explain_rows: Maximum rows passed to ``TreeExplainer`` per call.
                          Larger batches are sampled down to this limit to keep
                          latency bounded.  Default 500 rows.

    Raises:
        ImportError:  if ``shap`` is not installed.
        ValueError:   if ``reference_X`` is not 2-D, or ``feature_names``
                      length does not match ``reference_X`` columns.
    """

    def __init__(
        self,
        model: Any,
        reference_X: np.ndarray,
        feature_names: list[str],
        *,
        max_explain_rows: int = 500,
    ) -> None:
        try:
            import shap as _shap
        except ImportError as exc:
            raise ImportError(
                "shap is required for ShapDriftAttributor. "
                "Install it with: pip install shap"
            ) from exc

        if reference_X.ndim != 2:
            raise ValueError(f"reference_X must be 2-D, got shape {reference_X.shape}")
        if len(feature_names) != reference_X.shape[1]:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but "
                f"reference_X has {reference_X.shape[1]} columns"
            )

        self.feature_names = list(feature_names)
        self.max_explain_rows = max_explain_rows

        self._explainer = _shap.TreeExplainer(model)

        # Baseline: mean |SHAP| per feature on the reference distribution.
        # We cap reference rows at max_explain_rows for consistency with
        # the inference path - the baseline should be computed with the
        # same budget.
        ref_capped = self._sample(reference_X, rng_seed=0)
        baseline_shap = self._mean_abs_shap(ref_capped)
        self.baseline: dict[str, float] = dict(zip(self.feature_names, baseline_shap))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute(self, X: np.ndarray) -> dict[str, float]:
        """Return per-feature SHAP importance shift vs. the training baseline.

        The shift for feature ``f`` is::

            shift_f = mean_abs_shap_current_f - baseline_f

        A positive shift indicates the model is relying on that feature more
        than it did at training time - a useful proxy for "drift that matters
        for predictions" as opposed to drift in irrelevant distribution tails.

        Args:
            X: Current-batch feature matrix, shape ``(n_samples, n_features)``.
               Must have the same column count as the reference matrix.

        Returns:
            Dict mapping each feature name to its importance shift.  Values
            are unbounded; zero means no change from baseline.

        Raises:
            ValueError: if ``X`` has a different number of features than the
                        reference matrix used at construction.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {X.shape[1]} features but attributor expects "
                f"{len(self.feature_names)}"
            )

        X_capped = self._sample(X)
        current_shap = self._mean_abs_shap(X_capped)

        return {
            name: float(current_shap[i] - self.baseline[name])
            for i, name in enumerate(self.feature_names)
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample(self, X: np.ndarray, *, rng_seed: int = 42) -> np.ndarray:
        """Return up to ``max_explain_rows`` rows, sampled without replacement."""
        if X.shape[0] <= self.max_explain_rows:
            return X
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(X.shape[0], size=self.max_explain_rows, replace=False)
        return X[idx]

    def _mean_abs_shap(self, X: np.ndarray) -> np.ndarray:
        """Return mean absolute SHAP value per feature, shape ``(n_features,)``.

        shap output format varies by version and model type:
        - shap < 0.40 binary classifier: list of two arrays, each (n, d)
        - shap >= 0.40 binary classifier: single array (n, d, 2)
        - regression / single output: array (n, d)

        In all cases we use the positive-class (index 1) slice for binary
        classification, or the full array for single-output models.
        """
        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            # Legacy format: list[class0_array, class1_array]
            arr = np.asarray(shap_values[1])
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                # New format: (n_samples, n_features, n_classes) - take class 1.
                arr = arr[:, :, 1]
        # arr is now (n_samples, n_features)
        result: np.ndarray = np.abs(arr).mean(axis=0)
        return result
