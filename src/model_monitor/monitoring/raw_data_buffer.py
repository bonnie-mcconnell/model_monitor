"""Rolling buffer of recent labeled inference batches used for retraining.

Design rationale
----------------
The ``RetrainEvidenceBuffer`` tells the decision engine *when* to retrain -
it accumulates aggregated monitoring signals (accuracy, F1, drift, trust)
and fires once sufficient evidence of degradation exists.

But ``RetrainPipeline`` needs *raw* labeled examples - feature vectors and
ground-truth labels - not monitoring summaries.  This class bridges that gap.

``RawDataBuffer`` collects (X, y) pairs from every inference batch where
labels are available.  When a retrain is triggered, ``DefaultModelActionExecutor``
consumes this buffer as the training dataset for the candidate model.

Memory cap
----------
The buffer is capped at ``max_rows``.  When the cap is reached, the oldest
chunk is dropped (FIFO) before appending the new one.  The cap prevents
unbounded memory growth in long-running deployments where retraining is
infrequent.

Thread safety
-------------
The buffer is designed for single-threaded use within the asyncio event loop.
Concurrent writers would need an asyncio.Lock wrapping ``add_batch``; the
current callers (Predictor.predict_batch and simulation_loop) are
single-threaded so no lock is needed here.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class RawDataBuffer:
    """Accumulates labeled ``(X, y)`` pairs from recent inference batches.

    Consumed by ``DefaultModelActionExecutor`` when a retrain decision fires,
    providing the actual feature data and labels to train a candidate model on.
    Without this buffer, the retrain pipeline would have no real data to learn
    from - only monitoring-metric aggregates, which are not a valid training set.

    Args:
        max_rows:      Maximum number of rows retained across all chunks.
                       Oldest chunks are evicted FIFO when the cap is exceeded.
                       Default (50 000) is enough for a meaningful retrain without
                       significant memory pressure.
        feature_names: Column names for the feature matrix.  Set on first
                       ``add_batch`` call; enforced on subsequent calls so
                       the buffer always has a consistent schema.
    """

    def __init__(self, max_rows: int = 50_000) -> None:
        if max_rows < 1:
            raise ValueError(f"max_rows must be >= 1, got {max_rows}")
        self.max_rows = max_rows
        self._X_chunks: deque[np.ndarray] = deque()
        self._y_chunks: deque[np.ndarray] = deque()
        self._chunk_sizes: deque[int] = deque()
        self._total_rows: int = 0
        self.feature_names: list[str] = []

    def add_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Append a labeled batch.  Oldest chunks are evicted if cap is exceeded.

        Args:
            X:             Feature matrix, shape (n_samples, n_features).
            y:             Label vector, shape (n_samples,).
            feature_names: Column names for X.  Must match on subsequent calls.

        Raises:
            ValueError: if X and y have mismatched row counts, or if
                        feature_names differs from the buffer's established schema.
        """
        if X.shape[0] != len(y):
            raise ValueError(f"X has {X.shape[0]} rows but y has {len(y)} elements")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        # Establish or verify schema consistency.
        if not self.feature_names:
            self.feature_names = list(feature_names)
        elif list(feature_names) != self.feature_names:
            raise ValueError(
                f"Feature schema mismatch: expected {self.feature_names}, "
                f"got {feature_names}"
            )

        n = X.shape[0]
        self._X_chunks.append(X.astype(np.float64))
        self._y_chunks.append(np.asarray(y))
        self._chunk_sizes.append(n)
        self._total_rows += n

        # Evict oldest chunks until we are within the cap.
        while self._total_rows > self.max_rows and self._X_chunks:
            evicted = self._chunk_sizes.popleft()
            self._X_chunks.popleft()
            self._y_chunks.popleft()
            self._total_rows -= evicted

    def size(self) -> int:
        """Return the number of labeled rows currently held."""
        return self._total_rows

    def ready(self, min_samples: int) -> bool:
        """Return True if the buffer holds at least ``min_samples`` rows."""
        return self._total_rows >= min_samples

    def reset_schema(self, new_feature_names: list[str]) -> None:
        """Discard all buffered data and adopt a new feature schema.

        Called by ``Predictor.reload()`` after a model promotion that changes
        the feature set.  Keeping stale rows from the old schema would produce
        a training DataFrame with the wrong columns and silently corrupt the
        retrain.  Starting fresh is the correct trade-off: the next retrain
        will use only data collected under the new model, which is also the
        right data to train on.

        Logs a warning at the call site with the row count being discarded so
        operators can see in the audit log when a schema change caused data
        to be dropped.

        Args:
            new_feature_names: The feature names for the new model schema.
        """
        old_schema = self.feature_names
        old_rows = self._total_rows

        self._X_chunks.clear()
        self._y_chunks.clear()
        self._chunk_sizes.clear()
        self._total_rows = 0
        self.feature_names = list(new_feature_names)

        if old_rows > 0:
            log.warning(
                "raw_data_buffer_schema_reset",
                extra={
                    "discarded_rows": old_rows,
                    "old_schema": old_schema,
                    "new_schema": new_feature_names,
                },
            )

    def consume(self) -> pd.DataFrame:
        """Return all buffered data as a DataFrame and clear the buffer.

        The returned DataFrame has one column per feature (named according to
        ``feature_names``) plus a ``label`` column.  This is the format expected
        by ``train_model()`` in ``training/train.py``.

        Returns:
            DataFrame with columns [*feature_names, "label"], or an empty
            DataFrame when the buffer is empty.
        """
        if not self._X_chunks:
            return pd.DataFrame()

        X_all = np.vstack(list(self._X_chunks))
        y_all = np.concatenate(list(self._y_chunks))

        self._X_chunks.clear()
        self._y_chunks.clear()
        self._chunk_sizes.clear()
        self._total_rows = 0

        df = pd.DataFrame(X_all, columns=self.feature_names)
        df["label"] = y_all
        return df
