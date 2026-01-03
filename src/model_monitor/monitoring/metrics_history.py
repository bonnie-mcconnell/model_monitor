import json
import time
from pathlib import Path
from typing import List, Optional

from model_monitor.monitoring.types import MetricRecord, DecisionType


class MetricsHistory:
    """
    Persistent metrics history store.

    Stores one JSONL record per batch.
    Used for:
    - monitoring dashboards
    - promotion / rollback analysis
    - auditability
    - offline evaluation
    """

    def __init__(self, path: Path | str = "data/metrics/metrics_history.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        batch_id: str,
        n_samples: int,
        accuracy: float,
        f1: float,
        avg_confidence: float,
        drift_score: float,
        decision_latency_ms: float,
        action: DecisionType,
        reason: str,
        previous_model: Optional[str] = None,
        new_model: Optional[str] = None,
    ) -> None:
        record: MetricRecord = {
            "timestamp": float(time.time()),
            "batch_id": batch_id,
            "n_samples": int(n_samples),
            "accuracy": float(accuracy),
            "f1": float(f1),
            "avg_confidence": float(avg_confidence),
            "drift_score": float(drift_score),
            "decision_latency_ms": float(decision_latency_ms),
            "action": action,
            "reason": reason,
            "previous_model": previous_model,
            "new_model": new_model,
        }

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self) -> List[MetricRecord]:
        """
        Read all stored metric records.
        """
        if not self.path.exists():
            return []

        records: List[MetricRecord] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        return records
