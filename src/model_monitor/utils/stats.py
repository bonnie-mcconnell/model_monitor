import numpy as np


def moving_avg(x, window: int):
    """
    Compute simple moving average over a 1D array.
    """
    x = np.asarray(x)
    if window <= 0:
        raise ValueError("window must be > 0")
    if len(x) < window:
        return np.array([])

    return np.convolve(x, np.ones(window), "valid") / window


def entropy_from_labels(labels) -> float:
    """
    Compute Shannon entropy from discrete labels.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-9)))
