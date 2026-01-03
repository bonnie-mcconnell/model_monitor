# keep but treat as regulatorly/audit grade decision log
import json
import time
from pathlib import Path
from typing import List, TypedDict

from model_monitor.monitoring.types import DecisionType


class DecisionRecord(TypedDict):
    timestamp: float
    batch_index: int
    action: DecisionType
    reason: str
    f1: float
    f1_baseline: float
    drift_score: float
    model_version: str | None


class DecisionHistory:
    """
    Persistent store for operational decisions (retrain / rollback / reject).

    Stored as JSONL for auditability and dashboard use.
    """

    def __init__(self, path: Path | str = "data/decisions/decision_history.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        batch_index: int,
        action: DecisionType,
        reason: str,
        f1: float,
        f1_baseline: float,
        drift_score: float,
        model_version: str | None,
    ) -> None:
        record: DecisionRecord = {
            "timestamp": time.time(),
            "batch_index": batch_index,
            "action": action,
            "reason": reason,
            "f1": f1,
            "f1_baseline": f1_baseline,
            "drift_score": drift_score,
            "model_version": model_version,
        }

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> List[DecisionRecord]:
        if not self.path.exists():
            return []

        records: List[DecisionRecord] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        return records
