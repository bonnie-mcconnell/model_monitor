"""
Tests for RetrainEvidenceBuffer.

The buffer has two load-bearing properties:
1. ready() gates retrain triggering - wrong bounds mean retraining
   fires too early (noise triggers) or never (genuine degradation ignored)
2. retrain_key() provides idempotency - non-determinism here means
   duplicate retrains run on restart, wasting compute

Both are tested against exact expected values, not just "it works".
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer


def _add_summaries(buf: RetrainEvidenceBuffer, n: int) -> None:
    for i in range(n):
        buf.add_summary(
            accuracy=0.85 + i * 0.001,
            f1=0.83 + i * 0.001,
            drift_score=0.05,
            trust_score=0.80,
            timestamp=time.time() + i,
        )


# ---------------------------------------------------------------------------
# ready() and size()
# ---------------------------------------------------------------------------

def test_not_ready_before_min_samples() -> None:
    buf = RetrainEvidenceBuffer(min_samples=5)
    _add_summaries(buf, 4)
    assert buf.ready() is False


def test_ready_at_exactly_min_samples() -> None:
    buf = RetrainEvidenceBuffer(min_samples=5)
    _add_summaries(buf, 5)
    assert buf.ready() is True


def test_ready_above_min_samples() -> None:
    buf = RetrainEvidenceBuffer(min_samples=5)
    _add_summaries(buf, 10)
    assert buf.ready() is True


def test_size_tracks_additions() -> None:
    buf = RetrainEvidenceBuffer(min_samples=3)
    assert buf.size() == 0
    _add_summaries(buf, 2)
    assert buf.size() == 2


# ---------------------------------------------------------------------------
# consume()
# ---------------------------------------------------------------------------

def test_consume_returns_all_rows() -> None:
    buf = RetrainEvidenceBuffer(min_samples=3)
    _add_summaries(buf, 3)
    df = buf.consume()
    assert len(df) == 3


def test_consume_clears_buffer() -> None:
    buf = RetrainEvidenceBuffer(min_samples=3)
    _add_summaries(buf, 3)
    buf.consume()
    assert buf.size() == 0
    assert buf.ready() is False


def test_consume_returns_correct_columns() -> None:
    buf = RetrainEvidenceBuffer(min_samples=1)
    _add_summaries(buf, 1)
    df = buf.consume()
    assert set(df.columns) == {"accuracy", "f1", "drift_score", "trust_score", "timestamp"}


def test_consume_empty_buffer_returns_empty_dataframe() -> None:
    buf = RetrainEvidenceBuffer(min_samples=3)
    df = buf.consume()
    assert df.empty


# ---------------------------------------------------------------------------
# retrain_key() - determinism and idempotency
# ---------------------------------------------------------------------------

def test_retrain_key_is_deterministic() -> None:
    """
    The same DataFrame must produce the same key on every call.
    This is the idempotency guarantee: if a retrain crashes and restarts,
    re-hashing the same evidence window produces the same key, so the
    duplicate is correctly detected and skipped.
    """
    buf = RetrainEvidenceBuffer(min_samples=3)
    # Use fixed timestamps to make the DataFrame fully deterministic
    for i in range(3):
        buf.add_summary(
            accuracy=0.85,
            f1=0.83,
            drift_score=0.05,
            trust_score=0.80,
            timestamp=1000.0 + i,  # fixed, not time.time()
        )
    df = buf.consume()

    # Hash the same DataFrame twice - must be identical
    key_a = buf.retrain_key(df)
    key_b = buf.retrain_key(df)
    assert key_a == key_b


def test_retrain_key_differs_for_different_data() -> None:
    """Different evidence windows must produce different keys."""
    buf_a = RetrainEvidenceBuffer(min_samples=2)
    buf_a.add_summary(accuracy=0.80, f1=0.78, drift_score=0.05,
                      trust_score=0.75, timestamp=1000.0)
    buf_a.add_summary(accuracy=0.81, f1=0.79, drift_score=0.06,
                      trust_score=0.76, timestamp=1001.0)

    buf_b = RetrainEvidenceBuffer(min_samples=2)
    buf_b.add_summary(accuracy=0.90, f1=0.88, drift_score=0.02,
                      trust_score=0.92, timestamp=1000.0)
    buf_b.add_summary(accuracy=0.91, f1=0.89, drift_score=0.03,
                      trust_score=0.93, timestamp=1001.0)

    df_a = buf_a.consume()
    df_b = buf_b.consume()

    assert buf_a.retrain_key(df_a) != buf_b.retrain_key(df_b)


def test_retrain_key_is_64_char_hex() -> None:
    """SHA-256 hex digest is always exactly 64 characters."""
    buf = RetrainEvidenceBuffer(min_samples=1)
    _add_summaries(buf, 1)
    df = buf.consume()
    key = buf.retrain_key(df)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_retrain_key_raises_on_empty_dataframe() -> None:
    buf = RetrainEvidenceBuffer(min_samples=1)
    with pytest.raises(ValueError, match="empty"):
        buf.retrain_key(pd.DataFrame())
