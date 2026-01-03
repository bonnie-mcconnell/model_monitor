from __future__ import annotations

from collections import deque
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


class StreamSimulator:
    """
    Offline batch stream simulator with optional feature drift
    and delayed label release.

    Intended for experimentation and testing — not production inference.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        batch_size: int = 256,
        label_delay: int = 3,
        seed: int = 42,
    ) -> None:
        if "label" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'label' column")

        self.df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.batch_size = batch_size
        self.label_delay = label_delay

        self._cursor = 0
        self._step = 0
        self._label_queue: deque[np.ndarray] = deque()

    def _apply_feature_drift(
        self,
        X: pd.DataFrame,
        step: int,
    ) -> pd.DataFrame:
        """
        Apply simple synthetic drift for demonstration purposes.
        """
        if step > 20:
            X = X * 1.1

        return X

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            pd.DataFrame,
            Optional[np.ndarray],
            str,
        ]
    ]:
        while self._cursor < len(self.df):
            batch = self.df.iloc[
                self._cursor : self._cursor + self.batch_size
            ].copy()
            self._cursor += self.batch_size

            y = batch.pop("label").to_numpy()
            X = self._apply_feature_drift(batch, self._step)

            self._label_queue.append(y)
            released = None
            if len(self._label_queue) > self.label_delay:
                released = self._label_queue.popleft()

            batch_id = f"offline_batch_{self._step}"
            self._step += 1

            yield X, released, batch_id
