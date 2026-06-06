"""Tests for DataQualityMonitor.

Verifies null rate detection, out-of-range detection, schema consistency
checks, score computation, and interaction with the Predictor trust score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_monitor.monitoring.data_quality import DataQualityMonitor, DataQualityReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=n).astype(float),
            "income": rng.uniform(20_000, 200_000, size=n),
            "score": rng.uniform(0.0, 1.0, size=n),
        }
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_with_feature_names() -> None:
    monitor = DataQualityMonitor(["a", "b", "c"])
    assert monitor._feature_names == ["a", "b", "c"]


def test_construction_rejects_empty_feature_names() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        DataQualityMonitor([])


def test_construction_rejects_invalid_null_rate() -> None:
    with pytest.raises(ValueError, match="max_null_rate"):
        DataQualityMonitor(["a"], max_null_rate=0.0)


# ---------------------------------------------------------------------------
# Clean data
# ---------------------------------------------------------------------------


def test_perfect_data_scores_one() -> None:
    df = _clean_df()
    monitor = DataQualityMonitor(list(df.columns))
    report = monitor.evaluate(df)
    assert report.quality_score == pytest.approx(1.0)
    assert report.null_rate == 0.0
    assert report.schema_ok is True
    assert len(report.issues) == 0


def test_last_report_is_stored() -> None:
    df = _clean_df()
    monitor = DataQualityMonitor(list(df.columns))
    assert monitor.last_report is None
    monitor.evaluate(df)
    assert isinstance(monitor.last_report, DataQualityReport)


# ---------------------------------------------------------------------------
# Null rate detection
# ---------------------------------------------------------------------------


def test_high_null_rate_penalises_score() -> None:
    df = _clean_df()
    # Introduce 20% nulls across all columns
    mask = np.random.default_rng(1).random(df.shape) < 0.20
    df_null = df.mask(mask)

    monitor = DataQualityMonitor(list(df.columns), max_null_rate=0.05)
    report = monitor.evaluate(df_null)

    assert report.null_rate > 0.05
    assert report.quality_score < 1.0
    assert any("high_null_rate" in issue for issue in report.issues)


def test_null_rate_within_threshold_no_penalty() -> None:
    """Null rate below max_null_rate should not trigger an issue.

    4 NaNs out of 200 rows * 3 columns = 0.67% null rate, well below 5%.
    The null sub-score = max(0, 1 - 0.0067/0.05) ≈ 0.87.
    The overall quality score may not be 1.0 (there's a small smooth penalty),
    but no 'high_null_rate' issue should be reported.
    """
    df = _clean_df(200)
    # 2% nulls - below the 5% default threshold
    df.iloc[:4, 0] = np.nan

    monitor = DataQualityMonitor(list(df.columns), max_null_rate=0.05)
    report = monitor.evaluate(df)
    # No issue should be raised below threshold
    assert not any("high_null_rate" in issue for issue in report.issues)
    # Score should be close to 1.0 but with a small smooth penalty
    assert report.quality_score > 0.8


# ---------------------------------------------------------------------------
# Out-of-range detection
# ---------------------------------------------------------------------------


def test_out_of_range_values_penalise_score() -> None:
    df = _clean_df(100)
    df.iloc[5, 0] = -1.0  # age below 0 - out of range
    df.iloc[10, 2] = 1.5  # score above 1.0 - out of range

    bounds = {"age": (0.0, 120.0), "income": (0.0, 1_000_000.0), "score": (0.0, 1.0)}
    monitor = DataQualityMonitor(list(df.columns), feature_bounds=bounds)
    report = monitor.evaluate(df)

    assert report.out_of_range_rate > 0.0
    assert report.quality_score < 1.0
    assert any("out_of_range" in issue for issue in report.issues)


def test_no_bounds_configured_skips_range_check() -> None:
    df = _clean_df()
    df.iloc[0, 0] = -9999.0  # would be out of range if bounds were configured

    monitor = DataQualityMonitor(list(df.columns))  # no feature_bounds
    report = monitor.evaluate(df)
    assert report.out_of_range_rate == 0.0
    assert not any("out_of_range" in issue for issue in report.issues)


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------


def test_missing_column_sets_schema_ok_false() -> None:
    df = _clean_df()
    monitor = DataQualityMonitor(list(df.columns) + ["extra_feature"])
    report = monitor.evaluate(df)

    assert report.schema_ok is False
    assert any("missing_columns" in issue for issue in report.issues)
    assert report.quality_score < 1.0


def test_extra_column_sets_schema_ok_false() -> None:
    df = _clean_df()
    df["unexpected_col"] = 0.0
    monitor = DataQualityMonitor(["age", "income", "score"])
    report = monitor.evaluate(df)

    assert report.schema_ok is False
    assert any("unexpected_columns" in issue for issue in report.issues)


def test_correct_schema_ok_true() -> None:
    df = _clean_df()
    monitor = DataQualityMonitor(list(df.columns))
    report = monitor.evaluate(df)
    assert report.schema_ok is True


# ---------------------------------------------------------------------------
# Score properties
# ---------------------------------------------------------------------------


def test_quality_score_bounded_zero_one() -> None:
    """Quality score must always be in [0, 1] regardless of inputs."""
    df = _clean_df()
    mask = np.ones(df.shape, dtype=bool)
    df_all_null = df.mask(mask)

    monitor = DataQualityMonitor(list(df.columns) + ["missing"])
    report = monitor.evaluate(df_all_null)
    assert 0.0 <= report.quality_score <= 1.0


def test_multiple_issues_compound_penalty() -> None:
    """Null rate + schema issue together should score lower than each alone."""
    df = _clean_df(100)
    df.iloc[:20, 0] = np.nan  # high null rate

    monitor_null_only = DataQualityMonitor(list(df.columns), max_null_rate=0.05)
    report_null = monitor_null_only.evaluate(df)

    monitor_both = DataQualityMonitor(
        list(df.columns) + ["missing_feature"], max_null_rate=0.05
    )
    report_both = monitor_both.evaluate(df)

    assert report_both.quality_score <= report_null.quality_score
