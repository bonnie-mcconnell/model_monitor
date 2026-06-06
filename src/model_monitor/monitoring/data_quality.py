"""Data quality monitoring for incoming inference batches.

PSI and model performance metrics are lagging indicators: they detect
problems *after* a model has already made predictions on bad data.
Data quality monitoring is a leading indicator: it catches upstream
data pipeline failures before they corrupt metrics or trust scores.

Common production ML failures caught here but missed by drift detection:
  - NaN/null rates spiking (upstream pipeline dropped a join, sensor offline)
  - Feature values escaping physical bounds (negative age, probability > 1.0)
  - Schema changes (new column ordering, dropped features, type coercions)

DataQualityMonitor returns a scalar ``quality_score`` in [0, 1] that feeds
the trust score directly.  A score of 1.0 means no quality issues were
detected; 0.0 means all checks failed.  The score is the unweighted mean
across all enabled checks so every check contributes equally.

The score feeds into the trust score as the ``data_quality`` component,
replacing the less informative ``confidence`` component when quality
monitoring is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    """Immutable result of one data quality evaluation.

    Attributes:
        quality_score:   [0, 1] aggregate score - 1.0 is perfect.
        null_rate:       fraction of values that are NaN/null across all features.
        out_of_range_rate: fraction of features with values outside configured bounds.
        schema_ok:       True when the incoming schema matches the reference schema.
        issues:          Human-readable descriptions of any detected issues.
    """

    quality_score: float
    null_rate: float
    out_of_range_rate: float
    schema_ok: bool
    issues: tuple[str, ...] = field(default_factory=tuple)


class DataQualityMonitor:
    """Check each incoming batch for null rates, range violations, and schema drift.

    Args:
        feature_names:   Ordered list of expected feature names.  Used to detect
                         schema changes (missing columns, new columns, reordering).
        feature_bounds:  Optional dict mapping feature name → (min, max) bounds.
                         Values outside these bounds are counted as range violations.
                         If None, range checking is skipped.
        max_null_rate:   Maximum acceptable proportion of null values before the
                         null check contributes 0.0 to the quality score.
                         Defaults to 0.05 (5%).
        max_oor_rate:    Maximum acceptable proportion of out-of-range features
                         before the range check contributes 0.0.
                         Defaults to 0.02 (2%).

    The quality score is the mean of three sub-scores:
      null_score   = max(0, 1 - null_rate / max_null_rate)
      range_score  = max(0, 1 - oor_rate / max_oor_rate)   (1.0 if no bounds)
      schema_score = 1.0 if schema matches, 0.0 otherwise

    This produces a smooth [0, 1] penalty that degrades proportionally rather
    than clipping to 0 at the first sign of issues.
    """

    def __init__(
        self,
        feature_names: list[str],
        *,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
        max_null_rate: float = 0.05,
        max_oor_rate: float = 0.02,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if max_null_rate <= 0 or max_null_rate > 1:
            raise ValueError("max_null_rate must be in (0, 1]")
        if max_oor_rate <= 0 or max_oor_rate > 1:
            raise ValueError("max_oor_rate must be in (0, 1]")

        self._feature_names = list(feature_names)
        self._feature_set = set(feature_names)
        self._feature_bounds = feature_bounds or {}
        self._max_null_rate = max_null_rate
        self._max_oor_rate = max_oor_rate

        # Last report stored for dashboard access without re-evaluation.
        self.last_report: DataQualityReport | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, X: pd.DataFrame) -> DataQualityReport:
        """Evaluate data quality for one batch.

        Args:
            X: feature matrix for this batch.  Column names are used for
               schema comparison.

        Returns:
            DataQualityReport with quality_score in [0, 1].
        """
        issues: list[str] = []
        sub_scores: list[float] = []

        # ── 1. Null rate ────────────────────────────────────────────────
        null_rate = self._null_rate(X)
        null_score = max(0.0, 1.0 - null_rate / self._max_null_rate)
        sub_scores.append(null_score)
        if null_rate > self._max_null_rate:
            issues.append(
                f"high_null_rate: {null_rate:.1%} exceeds threshold {self._max_null_rate:.1%}"
            )

        # ── 2. Out-of-range rate ────────────────────────────────────────
        if self._feature_bounds:
            oor_rate = self._out_of_range_rate(X)
            range_score = max(0.0, 1.0 - oor_rate / self._max_oor_rate)
            sub_scores.append(range_score)
            if oor_rate > self._max_oor_rate:
                issues.append(f"out_of_range: {oor_rate:.1%} of features exceed bounds")
        else:
            # Range checking disabled - contribute full score so it doesn't
            # drag down the mean when no bounds are configured.
            sub_scores.append(1.0)
            oor_rate = 0.0

        # ── 3. Schema consistency ───────────────────────────────────────
        schema_ok, schema_issues = self._check_schema(X)
        schema_score = 1.0 if schema_ok else 0.0
        sub_scores.append(schema_score)
        issues.extend(schema_issues)

        quality_score = float(np.mean(sub_scores))

        report = DataQualityReport(
            quality_score=quality_score,
            null_rate=null_rate,
            out_of_range_rate=oor_rate,
            schema_ok=schema_ok,
            issues=tuple(issues),
        )
        self.last_report = report
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _null_rate(self, X: pd.DataFrame) -> float:
        """Return fraction of null values across all cells."""
        total = X.shape[0] * X.shape[1]
        if total == 0:
            return 0.0
        return float(X.isnull().sum().sum()) / total

    def _out_of_range_rate(self, X: pd.DataFrame) -> float:
        """Return fraction of configured features that have ANY out-of-range values."""
        if not self._feature_bounds:
            return 0.0

        n_features_checked = 0
        n_out_of_range = 0

        for feat, (lo, hi) in self._feature_bounds.items():
            if feat not in X.columns:
                continue
            n_features_checked += 1
            col = X[feat].dropna()
            if ((col < lo) | (col > hi)).any():
                n_out_of_range += 1

        if n_features_checked == 0:
            return 0.0
        return n_out_of_range / n_features_checked

    def _check_schema(self, X: pd.DataFrame) -> tuple[bool, list[str]]:
        """Compare incoming column names against the reference schema.

        Checks:
          - Missing columns (expected but not present)
          - Extra columns (present but not expected)

        Column ordering is intentionally not checked: downstream code
        reindexes columns before prediction so ordering is harmless.
        """
        incoming = set(X.columns)
        missing = self._feature_set - incoming
        extra = incoming - self._feature_set

        issues: list[str] = []
        if missing:
            issues.append(f"missing_columns: {sorted(missing)}")
        if extra:
            issues.append(f"unexpected_columns: {sorted(extra)}")

        return len(issues) == 0, issues
