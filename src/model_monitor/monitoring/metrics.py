import csv
from pathlib import Path
from collections import deque
from typing import Deque, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


METRICS_PATH = Path("metrics/performance.csv")


class MetricTracker:
    """
    Tracks rolling-window performance metrics and persists snapshots to disk.
    """

    def __init__(self, window: int = 100):
        self.window = window

        self.y_true: Deque[int] = deque(maxlen=window)
        self.y_pred: Deque[int] = deque(maxlen=window)
        self.conf: Deque[float] = deque(maxlen=window)

        self.batch_id: Optional[str] = None

    def update(self, preds, confs, y_true, batch_id: str) -> None:
        self.batch_id = batch_id

        self.y_true.extend(y_true)
        self.y_pred.extend(preds)
        self.conf.extend(confs)

        self._log_snapshot()

    def snapshot(self) -> Optional[dict]:
        if not self.y_true:
            return None

        y_t = np.asarray(self.y_true)
        y_p = np.asarray(self.y_pred)
        conf = np.asarray(self.conf)

        metrics = {
            "accuracy": accuracy_score(y_t, y_p),
            "f1": f1_score(y_t, y_p, zero_division=0),
            "entropy": float(np.mean(-conf * np.log(conf + 1e-12))),
        }

        # Log-loss is undefined if only one class is present
        if len(np.unique(y_t)) > 1:
            metrics["logloss"] = log_loss(y_t, conf)
        else:
            metrics["logloss"] = np.nan

        return metrics

    def _log_snapshot(self) -> None:
        snapshot = self.snapshot()
        if snapshot is None:
            return

        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not METRICS_PATH.exists()

        with METRICS_PATH.open("a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "batch_id",
                    "accuracy",
                    "f1",
                    "entropy",
                    "logloss",
                ])

            writer.writerow([
                self.batch_id,
                round(snapshot["accuracy"], 4),
                round(snapshot["f1"], 4),
                round(snapshot["entropy"], 6),
                (
                    round(snapshot["logloss"], 6)
                    if not np.isnan(snapshot["logloss"])
                    else ""
                ),
            ])
