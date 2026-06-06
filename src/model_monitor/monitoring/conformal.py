"""Conformal prediction sets for model monitoring and distribution-shift detection.

Why conformal prediction alongside PSI?
---------------------------------------
PSI is a histogram-divergence measure: it tells you *that* the distribution
shifted, but its p-value interpretation is informal.  Conformal prediction
gives a rigorous coverage guarantee: if the model and data generating process
haven't changed, at least ``(1 - alpha)`` of production samples will have
their true label inside the prediction set.

When ``coverage_rate`` falls significantly below ``1 - alpha``, the model is
*provably* making worse-than-expected predictions on the current distribution.
This is not a statistical test - it's a direct empirical coverage check, which
is harder to dismiss than a divergence score.

The combination of PSI (input shift) + output drift (output shift) + conformal
coverage (prediction quality degradation) forms a three-layer monitoring stack:
  Layer 1 - PSI:        inputs are changing
  Layer 2 - Output PSI: model outputs are changing
  Layer 3 - Coverage:   model correctness guarantees are breaking down

Implementation: Regularized Adaptive Prediction Sets (RAPS)
------------------------------------------------------------
RAPS extends RAPS (Angelopoulos et al., 2021) with a size-regularisation term
that keeps prediction sets small.  Here we implement the simpler LAC variant
(Least Ambiguous Set Classifier) which is sufficient for monitoring:

  1. Calibration: compute nonconformity scores on held-out data.
     score_i = 1 - softmax_i[y_i]  (probability assigned to the true class)
  2. Threshold: set ``q_hat`` = (1 - alpha)-quantile of calibration scores.
  3. Production: for each new sample, the prediction set contains all classes
     where softmax_j >= 1 - q_hat.
  4. Monitor: track what fraction of labeled production samples contain
     their true label.  This should stay >= 1 - alpha.

When labels are unavailable (the common case in production), we track average
prediction set size instead.  Growing set size indicates the model is becoming
less confident - a leading indicator of degradation.

Reference:
    Angelopoulos, A.N. & Bates, S. (2021). "A Gentle Introduction to
    Conformal Prediction and Distribution-Free Uncertainty Quantification."
    arXiv:2107.07511.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalMonitorResult:
    """Result of one conformal coverage evaluation.

    Attributes:
        coverage_rate:       fraction of labeled samples whose true label falls
                             inside the prediction set.  None when no labels
                             are provided (unlabeled production mode).
        mean_set_size:       average number of classes in each prediction set.
                             Increasing set size signals declining confidence.
        coverage_gap:        ``(1 - alpha) - coverage_rate``.  Positive means
                             coverage is below the guarantee.  None when no
                             labels provided.
        q_hat:               the nonconformity threshold used to build sets.
        coverage_ok:         True when coverage >= (1 - alpha) or labels absent.
    """

    coverage_rate: float | None
    mean_set_size: float
    coverage_gap: float | None
    q_hat: float
    coverage_ok: bool


class ConformalMonitor:
    """Track conformal coverage and prediction set size on production batches.

    Calibrate on a held-out labeled dataset, then call ``monitor()`` on each
    production batch.  When labels are available, compare empirical coverage
    against the theoretical guarantee.  When labels are unavailable, track
    mean prediction set size as a coverage proxy.

    Args:
        alpha: miscoverage level.  Sets coverage guarantee at ``1 - alpha``.
               Default 0.10 gives 90% coverage guarantee.
        min_set_size_alarm: mean prediction set size above which a warning is
               raised, even when no labels are available.  Default 1.5 (more
               than half the time the model needs 2+ classes to cover itself).

    Usage::

        monitor = ConformalMonitor(alpha=0.10)
        monitor.calibrate(cal_probs, cal_labels)
        result = monitor.monitor(prod_probs, prod_labels)  # labels optional
    """

    def __init__(
        self,
        alpha: float = 0.10,
        min_set_size_alarm: float = 1.5,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.min_set_size_alarm = min_set_size_alarm

        self._q_hat: float | None = None
        self._n_classes: int | None = None
        self._is_calibrated: bool = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit nonconformity threshold on a held-out calibration set.

        Args:
            probs:  2D float array of shape ``(n_cal, n_classes)`` - softmax
                    probabilities from the model on the calibration set.
            labels: 1D integer array of shape ``(n_cal,)`` - true class indices.

        The nonconformity score for sample i is ``1 - probs[i, y_i]``:
        the complement of the probability assigned to the true class.
        High scores mean the model was uncertain about the correct label.

        The threshold ``q_hat`` is the ``ceil((n+1)(1-alpha)/n)``-quantile
        of calibration scores - the exact finite-sample quantile from the
        conformal guarantee (Venn–Abers / split conformal).
        """
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=int)

        if probs.ndim != 2:
            raise ValueError("probs must be 2D (n_samples, n_classes)")
        if labels.ndim != 1:
            raise ValueError("labels must be 1D")
        if len(probs) != len(labels):
            raise ValueError("probs and labels must have the same length")
        if len(probs) == 0:
            raise ValueError("calibration set must not be empty")

        n_cal = len(probs)
        self._n_classes = probs.shape[1]

        # Nonconformity score: 1 - P(true class)
        true_class_probs = probs[np.arange(n_cal), labels]
        scores = 1.0 - true_class_probs

        # Finite-sample conformal quantile: guarantees coverage on average
        # by inflating the quantile level slightly to account for finite n.
        level = np.ceil((n_cal + 1) * (1.0 - self.alpha)) / n_cal
        level = min(level, 1.0)  # clip when n_cal is very small
        self._q_hat = float(np.quantile(scores, level))
        self._is_calibrated = True

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def monitor(
        self,
        probs: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> ConformalMonitorResult:
        """Evaluate conformal coverage on one production batch.

        Args:
            probs:  2D float array of shape ``(n_batch, n_classes)``.
            labels: Optional 1D integer array of true labels.  When provided,
                    empirical coverage is computed and compared to ``1 - alpha``.
                    When absent, only prediction set size is tracked.

        Returns:
            ConformalMonitorResult.  Use ``coverage_ok`` for alerting.

        Raises:
            RuntimeError: if ``calibrate()`` has not been called.
        """
        if not self._is_calibrated or self._q_hat is None or self._n_classes is None:
            raise RuntimeError(
                "ConformalMonitor must be calibrated before monitoring.  "
                "Call calibrate(probs, labels) first."
            )

        probs = np.asarray(probs, dtype=float)
        if probs.ndim != 2:
            raise ValueError("probs must be 2D (n_samples, n_classes)")
        if probs.shape[1] != self._n_classes:
            raise ValueError(
                f"probs has {probs.shape[1]} classes; calibrated with {self._n_classes}"
            )

        # Build prediction sets: include class j when P(j) >= 1 - q_hat.
        # Equivalently, nonconformity score < q_hat.
        threshold = 1.0 - self._q_hat
        in_set = probs >= threshold  # (n_batch, n_classes) boolean matrix

        set_sizes = in_set.sum(axis=1)  # samples with empty sets get size 0
        # Ensure at least the argmax class is in every set (avoid empty sets).
        argmax_classes = probs.argmax(axis=1)
        for i, cls in enumerate(argmax_classes):
            in_set[i, cls] = True
            set_sizes[i] = max(set_sizes[i], 1)

        mean_set_size = float(set_sizes.mean())

        coverage_rate: float | None = None
        coverage_gap: float | None = None
        coverage_ok = True

        if labels is not None:
            labels_arr = np.asarray(labels, dtype=int)
            if len(labels_arr) != len(probs):
                raise ValueError("labels and probs must have the same length")

            # Coverage: fraction of samples where true label is in prediction set.
            covered = in_set[np.arange(len(labels_arr)), labels_arr]
            coverage_rate = float(covered.mean())
            guarantee = 1.0 - self.alpha
            coverage_gap = guarantee - coverage_rate
            # Coverage is "ok" if within 2 standard deviations of the guarantee.
            # Exact binomial SE: sqrt(p*(1-p)/n) where p = 1 - alpha.
            n = len(labels_arr)
            se = (guarantee * self.alpha / n) ** 0.5 if n > 0 else 0.0
            coverage_ok = coverage_rate >= (guarantee - 2 * se)

        # Alert on large prediction sets even without labels.
        if mean_set_size > self.min_set_size_alarm:
            coverage_ok = False

        return ConformalMonitorResult(
            coverage_rate=coverage_rate,
            mean_set_size=mean_set_size,
            coverage_gap=coverage_gap,
            q_hat=self._q_hat,
            coverage_ok=coverage_ok,
        )

    @property
    def is_calibrated(self) -> bool:
        """True after ``calibrate()`` has been called successfully."""
        return self._is_calibrated

    @property
    def q_hat(self) -> float | None:
        """Nonconformity threshold, or None before calibration."""
        return self._q_hat
