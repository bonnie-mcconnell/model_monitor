from typing import Any, Dict, List, Optional
import pandas as pd

from model_monitor.monitoring.metrics_history import MetricsHistory
from model_monitor.monitoring.types import MetricRecord


class MetricsAnalytics:
    """
    Read-only analytics interface over metrics history.

    Supports pagination and lightweight filtering for dashboards and APIs.
    """

    def __init__(self, history: MetricsHistory | None = None) -> None:
        self.history = history or MetricsHistory()

    def _load_df(self) -> pd.DataFrame:
        rows: List[MetricRecord] = self.history.read_all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame.from_records(rows)

    def latest(self) -> Dict[str, Any]:
        df = self._load_df()
        if df.empty:
            return {}

        return {str(k): v for k, v in df.iloc[-1].to_dict().items()}

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        action: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Paginated list of metric records with optional filters.

        Args:
            limit: max records returned
            offset: starting index
            action: filter by decision action
            batch_id: filter by batch id
        """
        df = self._load_df()
        if df.empty:
            return []

        if action and "action" in df.columns:
            df = df[df["action"] == action]

        if batch_id and "batch_id" in df.columns:
            df = df[df["batch_id"] == batch_id]

        page = df.iloc[offset : offset + limit]

        return [
            {str(k): v for k, v in row.items()}
            for row in page.to_dict(orient="records")
        ]

    def summary(self) -> Dict[str, float]:
        """
        Aggregate means over key monitoring signals.
        """
        df = self._load_df()
        if df.empty:
            return {}

        numeric_cols = [
            "accuracy",
            "f1",
            "avg_confidence",
            "drift_score",
            "decision_latency_ms",
        ]

        available = [c for c in numeric_cols if c in df.columns]
        if not available:
            return {}

        return {
            str(k): float(v)
            for k, v in df[available].mean().to_dict().items()
        }

    def count(self, action: Optional[str] = None) -> int:
        """
        Count records, optionally filtered by action.
        """
        df = self._load_df()
        if df.empty:
            return 0

        if action and "action" in df.columns:
            df = df[df["action"] == action]

        return int(len(df))
