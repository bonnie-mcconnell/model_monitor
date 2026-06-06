"""Output (prediction) distribution drift monitor using PSI.

Feature-level PSI tells you *that* input data changed.  Output drift tells
you *that the model's predictions are shifting* - often detectable before
performance degrades, because the output distribution can shift even when
input PSI is still below threshold.

Common causes caught by output drift but missed by input PSI:
  - Class imbalance in predictions (model systematically preferring one class)
  - Score distribution compression (model becoming over-confident or under-confident)
  - Threshold sensitivity changes (small input shifts cause large output shifts)

OutputDriftMonitor mirrors the interface of DriftMonitor but operates on
predicted probability vectors rather than raw feature arrays.  A single
aggregate PSI score is returned: the average PSI across all output classes.
Per-class scores are stored on ``last_class_scores`` for dashboard inspection.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from model_monitor.monitoring.drift import compute_psi


class OutputDriftMonitor:
    """Track drift in a model's output probability distribution using PSI.

    Reference bin edges are fixed at construction time from the training-time
    probability distribution (``reference_probs``).  This mirrors the input
    DriftMonitor design: production PSI is always measured against the same
    bins used at training time, not recomputed from recent batches.

    Args:
        reference_probs: 2D array of shape ``(n_samples, n_classes)`` from
            the training or calibration set.  Used to compute reference bin
            edges for each output class.
        window: number of batches to accumulate before computing PSI.
            Mirrors ``DriftConfig.window`` for consistency.
        threshold: PSI threshold above which drift is considered significant.
            Matches the input drift threshold so operators have one consistent
            scale.

    Design note: why PSI for output drift?
    PSI is interpretable (0.1 / 0.25 thresholds are industry-standard),
    deterministic (same input always produces the same score), and the
    bin edges are stored at training time - the same property that makes it
    correct for input drift makes it correct here.
    """

    def __init__(
        self,
        reference_probs: np.ndarray,
        *,
        window: int = 5,
        threshold: float = 0.1,
    ) -> None:
        if reference_probs.ndim != 2:
            raise ValueError("reference_probs must be 2D (n_samples, n_classes)")
        if reference_probs.shape[0] == 0:
            raise ValueError("reference_probs must contain at least one sample")

        self.reference = reference_probs
        self.n_classes = reference_probs.shape[1]
        self.window = window
        self.threshold = threshold

        self._buffer: deque[np.ndarray] = deque(maxlen=window)

        # Per-class PSI from the most recent update() call.
        # Empty before the buffer window is filled.
        self.last_class_scores: list[float] = []

    def update(self, probs: np.ndarray) -> float:
        """Add a batch of predicted probabilities and return mean output PSI.

        Args:
            probs: 2D array of shape ``(n_batch, n_classes)`` - predicted
                probability vectors for this batch.  Must have the same number
                of classes as ``reference_probs``.

        Returns:
            Mean PSI across all output classes, or 0.0 when the buffer window
            has not yet been filled.
        """
        if probs.ndim != 2:
            raise ValueError("probs must be 2D (n_samples, n_classes)")
        if probs.shape[1] != self.n_classes:
            raise ValueError(
                f"probs has {probs.shape[1]} classes; expected {self.n_classes}"
            )

        self._buffer.append(probs)

        if len(self._buffer) < self.window:
            self.last_class_scores = []
            return 0.0

        recent = np.vstack(self._buffer)  # (window * batch_size, n_classes)

        scores = [
            compute_psi(self.reference[:, c], recent[:, c])
            for c in range(self.n_classes)
        ]
        self.last_class_scores = scores
        return float(np.mean(scores))

    @property
    def is_drifting(self) -> bool:
        """True when the most recent mean output PSI exceeds ``threshold``."""
        if not self.last_class_scores:
            return False
        return float(np.mean(self.last_class_scores)) >= self.threshold
