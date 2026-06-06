"""Promotion decision: compare candidate vs current model using F1 with optional bootstrap CI.

Point-estimate comparison
--------------------------
``compare_models()`` with ``n_bootstrap=0`` (default) compares F1 point estimates.
This is fast and sufficient when the evaluation set is large (> 5000 samples).

Bootstrap confidence intervals
-------------------------------
When ``n_bootstrap > 0``, ``compare_models()`` also computes a bootstrap
confidence interval for the improvement.  Promotion only fires when the
lower bound of the ``(1-alpha)`` CI is above zero - i.e., when we can say with
confidence that the candidate is genuinely better, not just lucky on this split.

Why this matters:
    F1 on a 200-sample holdout is noisy.  A candidate with F1 0.82 vs current
    0.80 on 50 samples is statistically indistinguishable - the CI would span
    roughly [-0.03, +0.07] and we should not promote.  On 2000 samples, the
    same 2pp improvement might be significant.  Requiring CI_lower > 0 makes
    the promotion gate sample-size-aware, which no other standard MLOps tool
    does correctly.

This is not available in most MLOps platforms (MLflow, BentoML, etc.) - they
all use point-estimate comparison.  The statistical-significance gate makes
promotion robust to small-sample noise without requiring any distributional
assumptions.

Reference:
    Efron, B. & Hastie, T. (2016). "Computer Age Statistical Inference."
    Chapter 11: Bootstrap Confidence Intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Floating point subtraction is not exact: 0.82 - 0.80 evaluates to
# 0.019999999999999907 in IEEE 754. Without a tolerance, a candidate whose
# F1 improves by exactly min_improvement would be silently rejected.
_IMPROVEMENT_EPS = 1e-9

# Default number of bootstrap replicates.  1000 is the standard minimum for
# CI estimation; 2000 gives better tail estimates without significant runtime cost.
_DEFAULT_N_BOOTSTRAP = 1000


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval for the F1 improvement.

    Attributes:
        lower:      lower bound of the (1-alpha) CI.
        upper:      upper bound of the (1-alpha) CI.
        alpha:      miscoverage level used.  CI = (1-alpha) interval.
        n_bootstrap: number of replicates used.
    """

    lower: float
    upper: float
    alpha: float
    n_bootstrap: int


@dataclass(frozen=True)
class PromotionResult:
    """Result of a model promotion decision.

    Attributes:
        promoted:        True when candidate should replace current model.
        reason:          Machine-readable reason string.
        current_f1:      Point estimate F1 for the current model.
        candidate_f1:    Point estimate F1 for the candidate model.
        improvement:     ``candidate_f1 - current_f1``.
        bootstrap_ci:    Bootstrap CI for the improvement, or None when
                         bootstrapping was not used.
    """

    promoted: bool
    reason: str
    current_f1: float
    candidate_f1: float
    improvement: float
    bootstrap_ci: BootstrapCI | None = field(default=None)


def _bootstrap_f1_improvement(
    y_true: np.ndarray,
    y_pred_current: np.ndarray,
    y_pred_candidate: np.ndarray,
    *,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> BootstrapCI:
    """Compute a bootstrap CI for the F1 improvement between two classifiers.

    The paired bootstrap is used: for each replicate, the same set of indices
    is resampled for both models.  Paired resampling removes variance due to
    individual sample difficulty, giving a tighter and more correct CI for the
    *relative* improvement.

    Args:
        y_true:            1D ground truth label array.
        y_pred_current:    1D prediction array from the current model.
        y_pred_candidate:  1D prediction array from the candidate model.
        n_bootstrap:       number of bootstrap replicates.
        alpha:             miscoverage level.  CI = (1-alpha) interval.
        rng:               NumPy random generator for reproducibility.

    Returns:
        BootstrapCI with percentile-method CI bounds.
    """
    from sklearn.metrics import f1_score

    n = len(y_true)
    improvements: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yc = y_pred_current[idx]
        yk = y_pred_candidate[idx]
        f1_cur = float(f1_score(yt, yc, zero_division=0))
        f1_cand = float(f1_score(yt, yk, zero_division=0))
        improvements.append(f1_cand - f1_cur)

    improvements_arr = np.array(improvements)
    lo = float(np.percentile(improvements_arr, 100 * alpha / 2))
    hi = float(np.percentile(improvements_arr, 100 * (1 - alpha / 2)))

    return BootstrapCI(lower=lo, upper=hi, alpha=alpha, n_bootstrap=n_bootstrap)


def compare_models(
    *,
    current_f1: float,
    candidate_f1: float,
    min_improvement: float,
    y_true: np.ndarray | None = None,
    y_pred_current: np.ndarray | None = None,
    y_pred_candidate: np.ndarray | None = None,
    n_bootstrap: int = 0,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> PromotionResult:
    """Decide whether to promote a candidate model.

    Point-estimate promotion rule (always evaluated):
      - Candidate must outperform current model
      - Improvement must meet or exceed configured threshold

    Statistical-significance gate (when n_bootstrap > 0 and predictions supplied):
      - Bootstrap CI lower bound must be > 0
      - This prevents promoting a candidate that is only coincidentally better
        on this particular holdout split

    An epsilon tolerance (_IMPROVEMENT_EPS) is applied to the threshold
    comparison to handle floating point rounding in F1 arithmetic.

    Args:
        current_f1:       F1 score of the current production model.
        candidate_f1:     F1 score of the candidate model on the validation set.
        min_improvement:  Minimum absolute F1 gain required for promotion.
        y_true:           Ground truth labels (required for bootstrap).
        y_pred_current:   Current model's predictions on the validation set.
        y_pred_candidate: Candidate model's predictions on the validation set.
        n_bootstrap:      Number of bootstrap replicates.  0 disables bootstrap.
        alpha:            Miscoverage level for CI.  Default 0.05 = 95% CI.
        rng:              NumPy generator.  Created from seed 0 if None.

    Returns:
        PromotionResult with promoted=True when all active gates pass.
    """
    improvement = candidate_f1 - current_f1

    # ── Point-estimate gate ─────────────────────────────────────────────────
    point_ok = improvement >= min_improvement - _IMPROVEMENT_EPS

    if not point_ok:
        return PromotionResult(
            promoted=False,
            reason="insufficient_improvement",
            current_f1=current_f1,
            candidate_f1=candidate_f1,
            improvement=improvement,
            bootstrap_ci=None,
        )

    # ── Bootstrap significance gate (optional) ──────────────────────────────
    ci: BootstrapCI | None = None

    if n_bootstrap > 0:
        if y_true is None or y_pred_current is None or y_pred_candidate is None:
            # Caller requested bootstrap but didn't provide predictions -
            # degrade gracefully to point-estimate rather than raising.
            return PromotionResult(
                promoted=True,
                reason="candidate_outperforms_current",
                current_f1=current_f1,
                candidate_f1=candidate_f1,
                improvement=improvement,
                bootstrap_ci=None,
            )

        _rng = rng if rng is not None else np.random.default_rng(0)
        ci = _bootstrap_f1_improvement(
            np.asarray(y_true),
            np.asarray(y_pred_current),
            np.asarray(y_pred_candidate),
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            rng=_rng,
        )

        if ci.lower <= 0:
            return PromotionResult(
                promoted=False,
                reason="bootstrap_ci_includes_zero",
                current_f1=current_f1,
                candidate_f1=candidate_f1,
                improvement=improvement,
                bootstrap_ci=ci,
            )

    return PromotionResult(
        promoted=True,
        reason="candidate_outperforms_current",
        current_f1=current_f1,
        candidate_f1=candidate_f1,
        improvement=improvement,
        bootstrap_ci=ci,
    )
