from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, TypedDict
from datetime import datetime, timezone

from model_monitor.core.decisions import DecisionType


class DecisionRecord(TypedDict):
    timestamp: float
    ts_iso: str
    batch_index: int
    action: DecisionType
    reason: str
    f1: float
    f1_baseline: float
    drift_score: float
    model_version: str | None


class DecisionHistory:
    """
    Persistent audit log for operational decisions.

    - Append-only
    - JSONL format
    - Human readable
    - Dashboard & compliance friendly
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
            "ts_iso": datetime.now(timezone.utc).isoformat(),
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
