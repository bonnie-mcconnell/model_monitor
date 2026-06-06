"""
Tests for monitoring/raw_data_buffer.py.

RawDataBuffer is the component that bridges the monitoring layer (which
detects *when* to retrain) and the training layer (which needs labeled data
to retrain *on*).  The critical properties tested here:

1. Schema consistency - once the first batch establishes feature names, every
   subsequent batch is validated against them.
2. FIFO eviction - the buffer respects max_rows by dropping oldest chunks.
3. Consume-and-clear - consume() returns all data and leaves the buffer empty.
4. ready() threshold - correct guard for min_samples.
5. Empty consume - returns an empty DataFrame with no crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_monitor.monitoring.raw_data_buffer import RawDataBuffer


def _make_batch(
    n: int = 100,
    n_features: int = 4,
    *,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    X = rng.normal(size=(n, n_features))
    y = rng.integers(0, 2, size=n)
    return X, y


FEATURE_NAMES = ["f0", "f1", "f2", "f3"]


# ---------------------------------------------------------------------------
# Basic add / size / ready
# ---------------------------------------------------------------------------


def test_buffer_starts_empty() -> None:
    buf = RawDataBuffer()
    assert buf.size() == 0
    assert not buf.ready(min_samples=1)


def test_add_batch_increments_size() -> None:
    buf = RawDataBuffer()
    X, y = _make_batch(100)
    buf.add_batch(X, y, FEATURE_NAMES)
    assert buf.size() == 100


def test_ready_false_below_threshold() -> None:
    buf = RawDataBuffer()
    X, y = _make_batch(50)
    buf.add_batch(X, y, FEATURE_NAMES)
    assert not buf.ready(min_samples=100)


def test_ready_true_at_threshold() -> None:
    buf = RawDataBuffer()
    X, y = _make_batch(100)
    buf.add_batch(X, y, FEATURE_NAMES)
    assert buf.ready(min_samples=100)


def test_ready_true_above_threshold() -> None:
    buf = RawDataBuffer()
    X, y = _make_batch(200)
    buf.add_batch(X, y, FEATURE_NAMES)
    assert buf.ready(min_samples=100)


# ---------------------------------------------------------------------------
# Schema enforcement
# ---------------------------------------------------------------------------


def test_feature_names_set_on_first_batch() -> None:
    buf = RawDataBuffer()
    X, y = _make_batch()
    buf.add_batch(X, y, FEATURE_NAMES)
    assert buf.feature_names == FEATURE_NAMES


def test_mismatched_feature_names_raises() -> None:
    buf = RawDataBuffer()
    X, y = _make_batch()
    buf.add_batch(X, y, FEATURE_NAMES)
    with pytest.raises(ValueError, match="schema mismatch"):
        buf.add_batch(X, y, ["a", "b", "c", "d"])


def test_wrong_X_shape_raises() -> None:
    buf = RawDataBuffer()
    X_1d = np.zeros(10)
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="2-D"):
        buf.add_batch(X_1d, y, FEATURE_NAMES)


def test_mismatched_X_y_rows_raises() -> None:
    buf = RawDataBuffer()
    X = np.zeros((10, 4))
    y = np.zeros(9, dtype=int)
    with pytest.raises(ValueError, match="rows"):
        buf.add_batch(X, y, FEATURE_NAMES)


# ---------------------------------------------------------------------------
# FIFO eviction
# ---------------------------------------------------------------------------


def test_fifo_eviction_respects_max_rows() -> None:
    """Oldest chunks are dropped when max_rows is exceeded."""
    buf = RawDataBuffer(max_rows=150)
    X, y = _make_batch(100)
    buf.add_batch(X, y, FEATURE_NAMES)  # 100 rows
    buf.add_batch(X, y, FEATURE_NAMES)  # 200 rows → evict first chunk
    assert buf.size() <= 150


def test_fifo_eviction_keeps_total_at_or_below_max() -> None:
    buf = RawDataBuffer(max_rows=300)
    for i in range(5):
        X, y = _make_batch(100, rng_seed=i)
        buf.add_batch(X, y, FEATURE_NAMES)
    assert buf.size() <= 300


def test_max_rows_one_raises() -> None:
    with pytest.raises(ValueError, match="max_rows"):
        RawDataBuffer(max_rows=0)


# ---------------------------------------------------------------------------
# consume
# ---------------------------------------------------------------------------


def test_consume_returns_dataframe_with_label_column() -> None:
    buf = RawDataBuffer()
    X, y = _make_batch(50)
    buf.add_batch(X, y, FEATURE_NAMES)
    df = buf.consume()
    assert isinstance(df, pd.DataFrame)
    assert "label" in df.columns
    for name in FEATURE_NAMES:
        assert name in df.columns


def test_consume_returns_correct_row_count() -> None:
    buf = RawDataBuffer()
    X, y = _make_batch(80)
    buf.add_batch(X, y, FEATURE_NAMES)
    df = buf.consume()
    assert len(df) == 80


def test_consume_clears_buffer() -> None:
    buf = RawDataBuffer()
    X, y = _make_batch(50)
    buf.add_batch(X, y, FEATURE_NAMES)
    buf.consume()
    assert buf.size() == 0
    assert not buf.ready(min_samples=1)


def test_consume_empty_returns_empty_dataframe() -> None:
    buf = RawDataBuffer()
    df = buf.consume()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_consume_accumulates_multiple_batches() -> None:
    """Data from multiple add_batch calls is all present after consume."""
    buf = RawDataBuffer()
    for i in range(3):
        X, y = _make_batch(40, rng_seed=i)
        buf.add_batch(X, y, FEATURE_NAMES)
    df = buf.consume()
    assert len(df) == 120


def test_consumed_dataframe_is_trainable() -> None:
    """DataFrame from consume() must satisfy train_model() requirements."""
    buf = RawDataBuffer()
    X, y = _make_batch(100)
    buf.add_batch(X, y, FEATURE_NAMES)
    df = buf.consume()
    from model_monitor.training.train import train_model

    model = train_model(df)
    assert model is not None


def test_consume_preserves_float64_dtype() -> None:
    """Feature values must be float64 for PSI computation."""
    buf = RawDataBuffer()
    X = np.ones((20, 4), dtype=np.float32)
    y = np.zeros(20, dtype=int)
    buf.add_batch(X, y, FEATURE_NAMES)
    df = buf.consume()
    for name in FEATURE_NAMES:
        assert df[name].dtype == np.float64


# ---------------------------------------------------------------------------
# reset_schema
# ---------------------------------------------------------------------------


def test_reset_schema_clears_rows() -> None:
    """reset_schema discards all buffered rows."""
    buf = RawDataBuffer()
    buf.add_batch(*_make_batch(50), FEATURE_NAMES)
    assert buf.size() == 50

    buf.reset_schema(["a", "b", "c", "d"])
    assert buf.size() == 0


def test_reset_schema_updates_feature_names() -> None:
    """After reset_schema, feature_names reflects the new schema."""
    buf = RawDataBuffer()
    buf.add_batch(*_make_batch(10), FEATURE_NAMES)

    new_names = ["x", "y"]
    buf.reset_schema(new_names)
    assert buf.feature_names == new_names


def test_reset_schema_allows_new_batches() -> None:
    """After reset_schema, add_batch accepts data under the new schema."""
    buf = RawDataBuffer()
    buf.add_batch(*_make_batch(10), FEATURE_NAMES)

    new_names = ["x", "y"]
    buf.reset_schema(new_names)

    X_new = np.ones((5, 2), dtype=np.float64)
    y_new = np.zeros(5, dtype=int)
    buf.add_batch(X_new, y_new, new_names)
    assert buf.size() == 5


def test_reset_schema_on_empty_buffer_is_silent() -> None:
    """reset_schema on an empty buffer logs nothing and sets the schema."""
    buf = RawDataBuffer()
    buf.reset_schema(["a", "b"])
    assert buf.feature_names == ["a", "b"]
    assert buf.size() == 0


def test_reset_schema_rejects_old_schema_after_reset() -> None:
    """After reset_schema, batches with the old schema raise ValueError."""
    buf = RawDataBuffer()
    buf.add_batch(*_make_batch(10), FEATURE_NAMES)
    buf.reset_schema(["x", "y"])

    # Old schema must now be rejected.
    with pytest.raises(ValueError, match="schema mismatch"):
        buf.add_batch(*_make_batch(5), FEATURE_NAMES)
