"""
Offline batch stream simulator for experimentation and integration testing.

``StreamSimulator`` iterates a DataFrame as a sequence of fixed-size batches,
optionally applying synthetic feature drift and releasing labels with a
configurable delay. It is the data source used by the simulation loop
(``scripts/simulation_loop.py``) and can be used directly in integration tests
to drive the inference pipeline without a live data feed.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


class StreamSimulator:
    """
    Offline batch stream simulator with optional feature drift and delayed
    label release.

    Iterating an instance yields ``(X, y_released, batch_id)`` tuples where:
    - ``X`` is the feature ``DataFrame`` for this batch (label column removed)
    - ``y_released`` is the label array released after ``label_delay`` batches,
      or ``None`` if labels are still in the delay queue
    - ``batch_id`` is a zero-indexed string identifier for the batch

    Label delay models real-world inference pipelines where ground-truth
    labels are not available until after the prediction is served.

    Feature drift is injected after batch ``drift_at_step`` by multiplying
    all features by ``drift_scale``. This is intentionally simple - it is
    sufficient to trigger PSI-based drift detection in integration tests.

    Args:
        df:           DataFrame with a ``label`` column.
        batch_size:   Rows per batch.
        label_delay:  Batches to wait before releasing labels.
        drift_at_step: Batch index at which synthetic drift begins.
        drift_scale:  Multiplicative factor applied to features after drift.
        seed:         Random seed for the initial shuffle.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        batch_size: int = 256,
        label_delay: int = 3,
        drift_at_step: int = 20,
        drift_scale: float = 1.1,
        seed: int = 42,
    ) -> None:
        if "label" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'label' column")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if label_delay < 0:
            raise ValueError(f"label_delay must be non-negative, got {label_delay}")

        self._df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        self._batch_size = batch_size
        self._label_delay = label_delay
        self._drift_at_step = drift_at_step
        self._drift_scale = drift_scale

        self._cursor = 0
        self._step = 0
        self._label_queue: deque[np.ndarray] = deque()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def n_batches(self) -> int:
        """Total number of complete batches available."""
        return len(self._df) // self._batch_size

    def __iter__(
        self,
    ) -> StreamSimulator:
        return self

    def __next__(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray | None, str]:
        if self._cursor >= len(self._df):
            raise StopIteration

        batch = self._df.iloc[
            self._cursor : self._cursor + self._batch_size
        ].copy()
        self._cursor += self._batch_size

        y = batch.pop("label").to_numpy()
        X = self._maybe_apply_drift(batch, self._step)

        self._label_queue.append(y)
        y_released: np.ndarray | None = None
        if len(self._label_queue) > self._label_delay:
            y_released = self._label_queue.popleft()

        batch_id = f"batch_{self._step:04d}"
        self._step += 1

        return X, y_released, batch_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_apply_drift(self, X: pd.DataFrame, step: int) -> pd.DataFrame:
        """
        Apply synthetic feature drift after ``drift_at_step``.

        A simple multiplicative scale is sufficient to shift the feature
        distribution enough to trigger PSI-based drift detection.
        """
        if step > self._drift_at_step:
            return X * self._drift_scale
        return X
