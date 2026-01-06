from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import pandas as pd


class RetrainEvidenceBuffer:
    """
    Buffers aggregated monitoring signals to determine when retraining
    should be triggered.

    NOTE:
    This buffer does not store raw features or labels.
    It accumulates evidence of sustained degradation and hands control
    to the retraining pipeline when sufficient signal is present.
    """

    def __init__(self, min_samples: int):
        self.min_samples = min_samples
        self._rows: Deque[Dict] = deque()

    def add_summary(
        self,
        *,
        accuracy: float,
        f1: float,
        drift_score: float,
        trust_score: float,
        timestamp: float,
    ) -> None:
        self._rows.append(
            {
                "accuracy": accuracy,
                "f1": f1,
                "drift_score": drift_score,
                "trust_score": trust_score,
                "timestamp": timestamp,
            }
        )

    def size(self) -> int:
        return len(self._rows)

    def ready(self) -> bool:
        """
        Returns True if sufficient evidence has accumulated
        to justify a retraining attempt.
        """
        return self.size() >= self.min_samples

    def consume(self) -> pd.DataFrame:
        """
        Consume and clear buffered evidence.

        Returns:
            pd.DataFrame: aggregated degradation evidence
        """
        if not self._rows:
            return pd.DataFrame()

        df = pd.DataFrame(list(self._rows))
        self._rows.clear()
        return df
