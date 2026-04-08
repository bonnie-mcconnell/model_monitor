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
