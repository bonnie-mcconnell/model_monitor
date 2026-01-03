import numpy as np
from collections import deque
from model_monitor.config.settings import DriftConfig


EPS = 1e-6


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI).

    Bin edges are derived from the expected (reference)
    distribution and reused for the actual distribution.
    """
    if expected.ndim != 1 or actual.ndim != 1:
        raise ValueError("PSI inputs must be 1D arrays")

    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.unique(np.percentile(expected, percentiles))

    if len(bin_edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_dist = exp_counts / (exp_counts.sum() + EPS)
    act_dist = act_counts / (act_counts.sum() + EPS)

    psi = np.sum(
        (act_dist - exp_dist)
        * np.log((act_dist + EPS) / (exp_dist + EPS))
    )

    return float(psi)


class DriftMonitor:
    """
    Tracks feature-level drift using PSI over a rolling window.
    """

    def __init__(
        self,
        reference_features: np.ndarray,
        config: DriftConfig,
    ) -> None:
        if reference_features.ndim != 2:
            raise ValueError("reference_features must be 2D")

        self.reference = reference_features
        self.window = config.window
        self.threshold = config.psi_threshold
        self.buffer = deque(maxlen=self.window)

    def update(self, X: np.ndarray) -> float:
        """
        Add a new batch and return mean PSI across features.
        """
        if X.ndim != 2:
            raise ValueError("Incoming batch must be 2D")

        self.buffer.extend(X)

        if len(self.buffer) < self.window:
            return 0.0

        recent = np.asarray(self.buffer)
        scores = []

        for i in range(self.reference.shape[1]):
            scores.append(
                compute_psi(self.reference[:, i], recent[:, i])
            )

        return float(np.mean(scores))
