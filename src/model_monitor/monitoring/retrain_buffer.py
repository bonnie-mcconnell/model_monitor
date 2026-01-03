from collections import deque
from typing import Deque, Sequence

import pandas as pd


class RetrainBuffer:
    """
    Accumulates labeled samples until retraining conditions are met.
    """

    def __init__(self, min_samples: int):
        self.min_samples = min_samples
        self._rows: Deque[pd.DataFrame] = deque()

    def add_batch(
        self,
        X,
        y,
        *,
        feature_names: Sequence[str],
    ) -> None:
        df = pd.DataFrame(X, columns=list(feature_names))
        df["label"] = y
        self._rows.append(df)

    def size(self) -> int:
        return sum(len(df) for df in self._rows)

    def ready(self) -> bool:
        return self.size() >= self.min_samples

    def consume(self) -> pd.DataFrame:
        """
        Consume and clear buffered data.

        Returns empty DataFrame if buffer is empty.
        """
        if not self._rows:
            return pd.DataFrame()

        data = pd.concat(self._rows, ignore_index=True)
        self._rows.clear()
        return data
