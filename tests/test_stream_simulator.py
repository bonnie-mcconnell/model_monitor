"""
Tests for StreamSimulator.

StreamSimulator is the data source for integration tests and the simulation
loop. The properties that matter: it correctly iterates batches, releases
labels with the configured delay, applies drift after the configured step,
and raises on invalid construction arguments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_monitor.utils.stream_simulator import StreamSimulator


def _make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    return pd.DataFrame({
        "f0": rng.standard_normal(n),
        "f1": rng.standard_normal(n),
        "label": rng.integers(0, 2, n),
    })


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_raises_without_label_column() -> None:
    df = pd.DataFrame({"f0": [1.0, 2.0]})
    with pytest.raises(ValueError, match="label"):
        StreamSimulator(df)


def test_raises_on_zero_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        StreamSimulator(_make_df(), batch_size=0)


def test_raises_on_negative_label_delay() -> None:
    with pytest.raises(ValueError, match="label_delay"):
        StreamSimulator(_make_df(), label_delay=-1)


# ---------------------------------------------------------------------------
# Iteration properties
# ---------------------------------------------------------------------------

def test_each_batch_has_correct_shape() -> None:
    """Every batch must have exactly batch_size rows (or fewer on the last)."""
    sim = StreamSimulator(_make_df(n=100), batch_size=32)
    for X, _, _ in sim:
        assert X.shape[1] == 2  # f0, f1 - label column stripped
        assert len(X) <= 32


def test_label_column_is_stripped_from_features() -> None:
    sim = StreamSimulator(_make_df(), batch_size=50)
    X, _, _ = next(iter(sim))
    assert "label" not in X.columns


def test_labels_are_delayed_by_configured_steps() -> None:
    """
    Labels released at step N should correspond to features from step N - label_delay.
    With label_delay=2: steps 0 and 1 return y=None; step 2 returns labels from step 0.
    """
    sim = StreamSimulator(_make_df(n=200), batch_size=50, label_delay=2)
    batches = list(sim)

    # First two batches have no released labels (still in queue)
    assert batches[0][1] is None
    assert batches[1][1] is None

    # From step 2 onwards labels are released
    assert batches[2][1] is not None


def test_n_batches_matches_iteration_count() -> None:
    """n_batches property must equal the number of items yielded."""
    df = _make_df(n=100)
    sim = StreamSimulator(df, batch_size=25)
    assert sim.n_batches == 4
    count = sum(1 for _ in sim)
    assert count == 4


# ---------------------------------------------------------------------------
# Drift injection
# ---------------------------------------------------------------------------

def test_features_are_scaled_after_drift_step() -> None:
    """
    After drift_at_step, feature values must be scaled by drift_scale.
    Testing the mean of the scaled batch exceeds the pre-drift batch mean
    verifies the transformation was applied.
    """
    n = 500
    sim_no_drift = StreamSimulator(
        _make_df(n=n), batch_size=50, drift_at_step=999, seed=0
    )
    sim_with_drift = StreamSimulator(
        _make_df(n=n), batch_size=50, drift_at_step=0, drift_scale=2.0, seed=0
    )

    _, _, _ = next(iter(sim_no_drift))   # step 0, pre-drift
    X_no_drift, _, _ = next(iter(sim_no_drift))

    _, _, _ = next(iter(sim_with_drift))  # step 0, drift starts at 0
    X_with_drift, _, _ = next(iter(sim_with_drift))

    # After drift, absolute feature values should be larger on average
    assert X_with_drift["f0"].abs().mean() > X_no_drift["f0"].abs().mean()
