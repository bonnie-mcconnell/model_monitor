"""Adaptive threshold calibration for trust score and drift monitoring.

The problem with static thresholds
------------------------------------
The default PSI thresholds (0.10 warn, 0.25 critical) and trust score
thresholds (0.60 warn, 0.40 critical) are industry-standard starting points.
They are not calibrated to any specific deployment.

A feature that is naturally high-variance (e.g., a volatile market indicator)
will routinely produce PSI values of 0.08–0.12 even on stable data.  With a
0.10 threshold this produces constant false-positive warnings that operators
learn to ignore - alert fatigue.

Conversely, a feature that is normally extremely stable (e.g., a binary flag)
might indicate a genuine pipeline failure with PSI = 0.05.  The static threshold
would miss it.

This module's approach: percentile-based calibration from stable reference windows
-----------------------------------------------------------------------------------
Observe the monitoring system on a known-stable evaluation window (e.g., the
first N batches after a clean deployment).  Record the PSI values produced by
each feature during this stable period.  The calibrated warn threshold is the
``(1-alpha)``-percentile of these stable-period PSI values.

    warn_threshold_i = percentile(stable_psi_i, 1 - alpha)

At significance level alpha=0.05, the threshold is set such that on stable
data, only 5% of batches would exceed it.  This is the correct false-positive
rate for a one-sided threshold.

Similarly for the trust score: the calibrated floor is the alpha-percentile of
stable-period trust scores.

Output
------
``ThresholdAdvisor.recommend()`` returns a ``ThresholdRecommendation`` with:
  - Per-feature recommended PSI warn threshold
  - Recommended trust score warn and critical thresholds
  - A summary of whether the defaults are appropriate or should be adjusted

This is the foundation of adaptive monitoring - thresholds that are calibrated
to the specific model and deployment rather than copied from a paper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ThresholdRecommendation:
    """Data-driven threshold recommendations from stable reference observations.

    Attributes:
        psi_warn_per_feature:  Recommended PSI warn threshold per feature.
                               Calibrated at the (1-alpha) percentile of
                               stable-period PSI values.
        psi_warn_global:       Single recommended global PSI warn threshold
                               (median of per-feature recommendations).
        trust_warn:            Recommended trust score warning threshold
                               (alpha percentile of stable-period trust scores).
        trust_critical:        Recommended trust score critical threshold
                               (alpha/2 percentile - tighter than warn).
        n_reference_batches:   Number of stable batches used for calibration.
        alpha:                 Significance level used.
        feature_names:         Feature names (same order as psi_warn_per_feature).
        notes:                 Human-readable notes on any unusual observations.
    """

    psi_warn_per_feature: tuple[float, ...]
    psi_warn_global: float
    trust_warn: float
    trust_critical: float
    n_reference_batches: int
    alpha: float
    feature_names: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


class ThresholdAdvisor:
    """Calibrate monitoring thresholds from stable reference observations.

    Args:
        feature_names:   Ordered feature names (must match PSI score ordering).
        alpha:           Desired false-positive rate for threshold exceedance.
                         Default 0.05 = 5% of stable-period batches exceed threshold.
        min_batches:     Minimum number of stable batches required before
                         recommendations are meaningful.  Default 30.

    Usage::

        advisor = ThresholdAdvisor(feature_names=["f0", "f1", "f2"])

        # During stable reference period, record each batch's signals:
        for batch in stable_batches:
            advisor.observe(
                psi_scores=batch.psi_per_feature,
                trust_score=batch.trust_score,
            )

        rec = advisor.recommend()
        print(rec.psi_warn_global)        # calibrated global PSI threshold
        print(rec.psi_warn_per_feature)   # per-feature PSI thresholds
        print(rec.notes)                  # any warnings about unusual variance
    """

    def __init__(
        self,
        feature_names: list[str],
        *,
        alpha: float = 0.05,
        min_batches: int = 30,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if min_batches < 10:
            raise ValueError(
                "min_batches must be at least 10 for meaningful percentiles"
            )

        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)
        self.alpha = alpha
        self.min_batches = min_batches

        # Accumulated observations: each row is one stable batch.
        self._psi_observations: list[list[float]] = []  # (n_batches, n_features)
        self._trust_observations: list[float] = []

    def observe(
        self,
        psi_scores: Sequence[float],
        trust_score: float,
    ) -> None:
        """Record one stable-period batch observation.

        Args:
            psi_scores:   Per-feature PSI scores (must match ``feature_names`` order).
            trust_score:  Composite trust score for this batch.
        """
        if len(psi_scores) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} PSI scores, got {len(psi_scores)}"
            )
        if not 0.0 <= trust_score <= 1.0:
            raise ValueError(f"trust_score must be in [0, 1], got {trust_score}")

        self._psi_observations.append(list(psi_scores))
        self._trust_observations.append(trust_score)

    def recommend(self) -> ThresholdRecommendation:
        """Produce data-driven threshold recommendations.

        Returns:
            ``ThresholdRecommendation`` with calibrated per-feature and global
            PSI thresholds, and trust score warn/critical thresholds.

        Raises:
            ValueError: if fewer than ``min_batches`` observations have been
                        recorded.  Recommendations from small samples are
                        unreliable.
        """
        n = len(self._trust_observations)
        if n < self.min_batches:
            raise ValueError(
                f"Need at least {self.min_batches} stable-period observations "
                f"for meaningful threshold calibration, have {n}. "
                f"Continue observing stable batches and retry."
            )

        psi_matrix = np.array(self._psi_observations)  # (n, n_features)
        trust_arr = np.array(self._trust_observations)  # (n,)

        # PSI thresholds: (1 - alpha) percentile per feature.
        # We want the threshold above which only alpha-fraction of stable batches fall.
        psi_per_feature = [
            float(np.percentile(psi_matrix[:, i], 100 * (1 - self.alpha)))
            for i in range(self.n_features)
        ]

        # Global PSI threshold: median of per-feature thresholds.
        # This gives an aggregate threshold that balances across features.
        psi_global = float(np.median(psi_per_feature))

        # Trust score thresholds: lower-tail percentiles.
        trust_warn = float(np.percentile(trust_arr, 100 * self.alpha))
        trust_critical = float(np.percentile(trust_arr, 100 * self.alpha / 2))

        notes = self._generate_notes(psi_per_feature, psi_global, trust_arr)

        return ThresholdRecommendation(
            psi_warn_per_feature=tuple(round(t, 4) for t in psi_per_feature),
            psi_warn_global=round(psi_global, 4),
            trust_warn=round(trust_warn, 4),
            trust_critical=round(trust_critical, 4),
            n_reference_batches=n,
            alpha=self.alpha,
            feature_names=tuple(self.feature_names),
            notes=tuple(notes),
        )

    def _generate_notes(
        self,
        psi_per_feature: list[float],
        psi_global: float,
        trust_arr: np.ndarray,
    ) -> list[str]:
        notes: list[str] = []

        # Flag high-variance features (recommended threshold much above default 0.1)
        for name, threshold in zip(self.feature_names, psi_per_feature):
            if threshold > 0.25:
                notes.append(
                    f"Feature '{name}' is naturally high-variance "
                    f"(recommended PSI threshold {threshold:.3f} >> default 0.10). "
                    f"The default threshold will produce frequent false positives."
                )
            elif threshold < 0.02:
                notes.append(
                    f"Feature '{name}' is extremely stable in the reference period "
                    f"(recommended PSI threshold {threshold:.3f}). "
                    f"Consider tightening monitoring for this feature."
                )

        if psi_global > 0.15:
            notes.append(
                f"Global recommended PSI threshold ({psi_global:.3f}) exceeds the "
                f"default (0.10). The default is likely too sensitive for this "
                f"deployment - consider updating psi_threshold in drift.yaml."
            )
        elif psi_global < 0.05:
            notes.append(
                f"Global recommended PSI threshold ({psi_global:.3f}) is below the "
                f"default (0.10). The deployment data is unusually stable - "
                f"the default threshold may miss early drift signals."
            )

        trust_std = float(trust_arr.std())
        if trust_std > 0.10:
            notes.append(
                f"Trust score variance is high (std={trust_std:.3f}) during the "
                f"stable reference period. Consider investigating whether the "
                f"reference period is genuinely stable."
            )

        return notes

    @property
    def n_observations(self) -> int:
        """Number of stable-period observations recorded so far."""
        return len(self._trust_observations)

    @property
    def is_ready(self) -> bool:
        """True when enough observations have been recorded for recommendations."""
        return len(self._trust_observations) >= self.min_batches
