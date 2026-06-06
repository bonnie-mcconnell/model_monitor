"""PSI drift detection: compute_psi and DriftMonitor."""

from __future__ import annotations

from collections import deque

import numpy as np

from model_monitor.config.settings import DriftConfig

EPS = 1e-6


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    *,
    bin_edges: np.ndarray | None = None,
) -> float:
    """
    Population Stability Index (PSI).

    Bin edges are derived from the expected (reference) distribution
    and reused for the actual distribution.  When ``bin_edges`` is
    supplied (loaded from ``reference_stats.json`` at startup), those
    edges are used directly so production PSI is always measured against
    the *same* buckets that were computed at training time.  This is the
    property that makes PSI a valid drift signal: the reference and
    production histograms are built on a common scale.

    Args:
        expected:  reference distribution (1-D array from training data).
        actual:    production distribution (1-D array from a recent batch).
        bins:      number of equal-frequency bins to use when ``bin_edges``
                   is not supplied.  Ignored when ``bin_edges`` is given.
        bin_edges: pre-computed bin edges from ``reference_stats.json``.
                   Pass these to avoid recomputing from the reference array
                   at every call.  Must have at least 2 elements.
    """
    if expected.ndim != 1 or actual.ndim != 1:
        raise ValueError("PSI inputs must be 1D arrays")

    if bin_edges is not None:
        edges = bin_edges
    else:
        percentiles = np.linspace(0, 100, bins + 1)
        edges = np.unique(np.percentile(expected, percentiles))

    if len(edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_dist = exp_counts / (exp_counts.sum() + EPS)
    act_dist = act_counts / (act_counts.sum() + EPS)

    psi = np.sum((act_dist - exp_dist) * np.log((act_dist + EPS) / (exp_dist + EPS)))
    return float(psi)


class DriftMonitor:
    """
    Tracks feature-level drift using PSI over a rolling window.

    When ``stored_bin_edges`` is supplied (a dict mapping feature index
    to pre-computed edges from ``reference_stats.json``), PSI is computed
    against those fixed edges.  This preserves the invariant described in
    the architecture docs: bin edges are determined once at training time
    and never recomputed from production data.
    """

    def __init__(
        self,
        reference_features: np.ndarray,
        config: DriftConfig,
        *,
        stored_bin_edges: dict[int, np.ndarray] | None = None,
    ) -> None:
        if reference_features.ndim != 2:
            raise ValueError("reference_features must be 2D")

        self.reference = reference_features
        self.window: int = config.window
        self.threshold: float = config.psi_threshold
        self.buffer: deque[np.ndarray] = deque(maxlen=self.window)
        # Per-feature bin edges loaded from reference_stats.json.
        # None means edges are recomputed from self.reference on each call
        # (original behaviour, preserved for backward compatibility when
        # reference_stats.json was produced by an older train.py).
        self._stored_bin_edges = stored_bin_edges or {}

        # Per-feature PSI scores from the most recent update().
        # Empty list before the buffer window is filled.
        self.last_feature_scores: list[float] = []

    def update(self, X: np.ndarray) -> float:
        """
        Add a new batch and return mean PSI across features.

        Per-feature scores are stored on ``last_feature_scores`` after every
        call so callers can inspect which features are drifting without
        re-running PSI.  The return value is always the scalar mean so
        existing callers need no changes.
        """
        if X.ndim != 2:
            raise ValueError("Incoming batch must be 2D")

        self.buffer.append(X)

        if len(self.buffer) < self.window:
            self.last_feature_scores = []
            return 0.0

        recent = np.vstack(self.buffer)
        scores = [
            compute_psi(
                self.reference[:, i],
                recent[:, i],
                bin_edges=self._stored_bin_edges.get(i),
            )
            for i in range(self.reference.shape[1])
        ]

        self.last_feature_scores = scores
        return float(np.mean(scores))
