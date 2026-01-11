from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Any

import hashlib
import pandas as pd
import numpy as np


class RetrainEvidenceBuffer:
    """
    Buffers aggregated monitoring signals to determine when retraining
    should be triggered.

    NOTE:
    - Stores only aggregated metrics, never raw features or labels
    - Responsible for producing deterministic retrain fingerprints
    """

    def __init__(self, min_samples: int) -> None:
        self.min_samples = min_samples
        self._rows: Deque[Dict[str, Any]] = deque()

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

    def retrain_key(self, df: pd.DataFrame) -> str:
        """
        Deterministic fingerprint for retrain idempotency.

        This key uniquely identifies the evidence window that
        triggered a retrain, ensuring crash-safe idempotency.
        """
        if df.empty:
            raise ValueError("Cannot compute retrain key for empty DataFrame")

        # Ensure deterministic ordering
        df_sorted = (
            df.sort_index(axis=1)
              .sort_values(by=list(df.columns))
              .reset_index(drop=True)
        )

        # Pandas → NumPy for stable hashing
        hashed = pd.util.hash_pandas_object(df_sorted, index=True)
        payload = hashed.to_numpy(dtype=np.uint64)

        return hashlib.sha256(payload.tobytes()).hexdigest()
