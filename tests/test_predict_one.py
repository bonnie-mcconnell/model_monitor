"""Tests for Monitor.predict_one() and Monitor.flush().

predict_one() is the streaming / per-request inference path.  It accumulates
individual rows into an internal buffer and flushes them through the full
predict() pipeline once ``flush_every`` rows have been collected.  This lets
real-time workloads (REST endpoints, event streams) get model predictions at
per-row latency while monitoring runs asynchronously on mini-batches.

Tests verify:

  - predict_one() returns the correct shape immediately (before flush)
  - Predictions match direct model.predict_proba output
  - Buffer flushes automatically after flush_every rows
  - Manual flush() drains a partial buffer
  - flush() on an empty buffer returns None
  - Flushed BatchResult has sane trust/drift bounds
  - Pending rows survive save()/load() round-trip
  - reset_after_retrain() discards the pending buffer
  - ValueError on 2-D input
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from model_monitor import Monitor, MonitorConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clf_and_ref() -> tuple[RandomForestClassifier, np.ndarray]:
    """Tiny trained classifier plus reference data."""
    X, y = make_classification(
        n_samples=600,
        n_features=6,
        n_informative=4,
        random_state=42,
    )
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X[:400], y[:400])
    return clf, X[400:]


@pytest.fixture()
def monitor(clf_and_ref: tuple[RandomForestClassifier, np.ndarray]) -> Monitor:
    """Monitor wrapping the small classifier; MMD and CUSUM disabled for speed."""
    clf, ref = clf_and_ref
    cfg = MonitorConfig(enable_mmd=False, cusum_delta=0.0)
    return Monitor(clf, reference_data=ref[:100], config=cfg)


# ---------------------------------------------------------------------------
# Return shape and correctness
# ---------------------------------------------------------------------------


def test_predict_one_returns_proba_vector(
    monitor: Monitor,
    clf_and_ref: tuple[RandomForestClassifier, np.ndarray],
) -> None:
    """Return value is the probability vector for the single row."""
    _, ref = clf_and_ref
    row = ref[0]
    result = monitor.predict_one(row, flush_every=999)
    # RandomForest with 2 classes → (2,) vector
    assert result.ndim == 1
    assert result.shape[0] == 2
    assert 0.0 <= result.sum() <= 1.0 + 1e-9


def test_predict_one_matches_model_proba(
    monitor: Monitor,
    clf_and_ref: tuple[RandomForestClassifier, np.ndarray],
) -> None:
    """predict_one output matches model.predict_proba for the same row."""
    clf, ref = clf_and_ref
    row = ref[5]
    got = monitor.predict_one(row, flush_every=999)
    expected = clf.predict_proba(row.reshape(1, -1))[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Buffer accumulation and auto-flush
# ---------------------------------------------------------------------------


def test_buffer_grows_before_flush(
    monitor: Monitor,
    clf_and_ref: tuple[RandomForestClassifier, np.ndarray],
) -> None:
    """Rows accumulate in _pending_rows until flush_every is reached."""
    _, ref = clf_and_ref
    for i in range(5):
        monitor.predict_one(ref[i], flush_every=10)
    assert len(monitor._pending_rows) == 5
    assert monitor.n_batches == 0  # no batch recorded yet


def test_auto_flush_at_threshold(
    monitor: Monitor,
    clf_and_ref: tuple[RandomForestClassifier, np.ndarray],
) -> None:
    """Buffer flushes and a batch is recorded when flush_every is reached."""
    _, ref = clf_and_ref
    flush_every = 8
    for i in range(flush_every):
        monitor.predict_one(ref[i], flush_every=flush_every)

    # After exactly flush_every calls the buffer should be empty and one
    # batch recorded.
    assert len(monitor._pending_rows) == 0
    assert monitor.n_batches == 1


def test_auto_flush_clears_labels_too(
    monitor: Monitor,
    clf_and_ref: tuple[RandomForestClassifier, np.ndarray],
) -> None:
    """Accumulated labels are cleared alongside rows on auto-flush."""
    clf, ref = clf_and_ref
    # Synthetic integer labels
    X, y = make_classification(n_samples=100, n_features=6, n_informative=4, random_state=42)
    m = Monitor(
        RandomForestClassifier(n_estimators=5, random_state=0).fit(X[:60], y[:60]),
        reference_data=X[60:],
        config=MonitorConfig(enable_mmd=False, cusum_delta=0.0),
    )
    for i in range(16):
        m.predict_one(X[i], y_true=int(y[i]), flush_every=16)

    assert len(m._pending_rows) == 0
    assert len(m._pending_labels) == 0


# ---------------------------------------------------------------------------
# Manual flush
# ---------------------------------------------------------------------------


def test_flush_drains_partial_buffer(
    monitor: Monitor,
    clf_and_ref: tuple[RandomForestClassifier, np.ndarray],
) -> None:
    """flush() processes whatever rows are buffered, even if < flush_every."""
    _, ref = clf_and_ref
    for i in range(3):
        monitor.predict_one(ref[i], flush_every=100)

    assert len(monitor._pending_rows) == 3
    result = monitor.flush()

    assert result is not None
    assert 0.0 <= result.trust_score <= 1.0
    assert monitor.n_batches == 1
    assert len(monitor._pending_rows) == 0


def test_flush_empty_buffer_returns_none(monitor: Monitor) -> None:
    """flush() on an empty buffer returns None without raising."""
    result = monitor.flush()
    assert result is None
    assert monitor.n_batches == 0


def test_flush_result_has_correct_batch_size(
    monitor: Monitor,
    clf_and_ref: tuple[RandomForestClassifier, np.ndarray],
) -> None:
    """BatchResult.predictions has length equal to the number of buffered rows."""
    _, ref = clf_and_ref
    n = 12
    for i in range(n):
        monitor.predict_one(ref[i], flush_every=999)
    result = monitor.flush()
    assert result is not None
    assert len(result.predictions) == n


# ---------------------------------------------------------------------------
# BatchResult contract after flush
# ---------------------------------------------------------------------------


def test_flushed_batch_result_bounds(
    monitor: Monitor,
    clf_and_ref: tuple[RandomForestClassifier, np.ndarray],
) -> None:
    """Trust and drift scores from a flushed batch are in [0, 1]."""
    _, ref = clf_and_ref
    for i in range(20):
        monitor.predict_one(ref[i], flush_every=999)
    result = monitor.flush()
    assert result is not None
    assert 0.0 <= result.trust_score <= 1.0
    assert 0.0 <= result.drift_score <= 1.0


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_pending_rows_survive_save_load(
    monitor: Monitor,
    clf_and_ref: tuple[RandomForestClassifier, np.ndarray],
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Pending rows are serialised by save() and restored by load()."""
    _, ref = clf_and_ref
    n_pending = 7
    for i in range(n_pending):
        monitor.predict_one(ref[i], flush_every=999)

    save_path = str(tmp_path / "monitor_state.json")  # type: ignore[operator]
    monitor.save(save_path)

    restored = Monitor.load(
        save_path,
        monitor._model,
        reference_data=ref[:100],
    )
    assert len(restored._pending_rows) == n_pending
    np.testing.assert_allclose(
        restored._pending_rows[0], monitor._pending_rows[0], rtol=1e-6
    )


# ---------------------------------------------------------------------------
# reset_after_retrain discards buffer
# ---------------------------------------------------------------------------


def test_reset_discards_pending_buffer(
    monitor: Monitor,
    clf_and_ref: tuple[RandomForestClassifier, np.ndarray],
) -> None:
    """reset_after_retrain() clears _pending_rows and _pending_labels.

    Pre-retrain rows belong to the old distribution and should not pollute
    the first post-retrain monitoring window.
    """
    _, ref = clf_and_ref
    for i in range(10):
        monitor.predict_one(ref[i], flush_every=999)
    assert len(monitor._pending_rows) == 10

    monitor.reset_after_retrain()

    assert len(monitor._pending_rows) == 0
    assert len(monitor._pending_labels) == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_predict_one_rejects_2d_input(monitor: Monitor) -> None:
    """predict_one raises ValueError when given a 2-D array."""
    X_2d = np.zeros((3, 6))
    with pytest.raises(ValueError, match="1-D feature vector"):
        monitor.predict_one(X_2d)  # noqa: E501  - passing wrong type intentionally


def test_predict_one_accepts_pandas_series(
    monitor: Monitor,
    clf_and_ref: tuple[RandomForestClassifier, np.ndarray],
) -> None:
    """predict_one works with a pandas.Series input."""
    import pandas as pd

    _, ref = clf_and_ref
    series = pd.Series(ref[0], dtype=float)
    result = monitor.predict_one(series, flush_every=999)
    assert result.ndim == 1
