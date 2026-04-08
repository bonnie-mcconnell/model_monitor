"""Evidence buffer: accumulates monitoring signals until ready to retrain."""
from __future__ import annotations

import hashlib
from collections import deque
from typing import Any

import numpy as np
import pandas as pd


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
        self._rows: deque[dict[str, Any]] = deque()

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

        This key uniquely identifies the evidence window that triggered a
        retrain, ensuring crash-safe deduplication across restarts.

        Implementation: sorts columns and rows for determinism, then hashes
        the raw float64 values via SHA-256. We avoid pd.util.hash_pandas_object
        because it is not documented as a stable public API and its output
        can change across pandas versions - which would silently invalidate
        stored keys and allow duplicate retrains after an upgrade.
        """
        if df.empty:
            raise ValueError("Cannot compute retrain key for empty DataFrame")

        # Sort columns and rows for determinism across different insertion orders
        df_sorted = (
            df.sort_index(axis=1)
              .sort_values(by=list(df.columns))
              .reset_index(drop=True)
        )

        # Hash raw float64 values - stable across pandas versions
        payload = df_sorted.to_numpy(dtype=np.float64).tobytes()
        return hashlib.sha256(payload).hexdigest()
