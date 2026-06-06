"""Statistical utilities: moving average, Shannon entropy, cosine similarity."""

from __future__ import annotations

import numpy as np


def moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    """
    Simple moving average over a 1-D array.

    Returns an empty array if len(x) < window, preserving the invariant
    that all returned values are computed over a full window.
    """
    x = np.asarray(x)
    if window <= 0:
        raise ValueError("window must be > 0")
    if len(x) < window:
        return np.array([])

    return np.convolve(x, np.ones(window), "valid") / window


def entropy_from_labels(labels: np.ndarray) -> float:
    """
    Shannon entropy of a discrete label distribution.

    Returns 0.0 for an empty input or a perfectly concentrated distribution.

    The 1e-9 additive smoothing prevents log(0) for zero-probability classes,
    but introduces a tiny negative bias for pure distributions
    (e.g. -sum([1.0 * log(1.0 + 1e-9)]) = -9.99e-10). The result is clamped
    to 0.0 to preserve the mathematical invariant that entropy is non-negative.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(max(0.0, -np.sum(probs * np.log(probs + 1e-9))))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.

    Returns 0.0 when either vector is the zero vector to avoid undefined
    division. Range is [-1.0, 1.0]; sentence embeddings are typically [0.0, 1.0].
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Measures how well a model's confidence scores match its actual accuracy.
    A perfectly calibrated model has ECE = 0: when it says 80% confidence,
    it is correct 80% of the time.

    ECE is important alongside F1 and accuracy because a model can achieve
    high accuracy while being systematically overconfident - which matters
    for any downstream decision that uses confidence as a threshold.

    Args:
        confidences: 1-D array of per-sample confidence scores in [0, 1].
                     For a classifier, this is the max class probability.
        correct:     1-D boolean array, True where the prediction was correct.
        n_bins:      Number of equal-width bins across [0, 1].

    Returns:
        ECE in [0, 1].  Lower is better.  Returns 0.0 for empty input.

    Reference:
        Guo et al. (2017) - "On Calibration of Modern Neural Networks"
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)

    if confidences.size == 0:
        return 0.0

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include the upper edge in the last bin to avoid dropping conf=1.0
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if not mask.any():
            continue

        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correct[mask].mean())
        bin_frac = float(mask.sum()) / confidences.size

        ece += bin_frac * abs(bin_acc - bin_conf)

    return float(ece)
