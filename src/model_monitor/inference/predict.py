from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple, Any

import joblib # type: ignore
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score # type: ignore

from model_monitor.config.settings import AppConfig
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import Decision
from model_monitor.monitoring.drift import DriftMonitor
from model_monitor.core.decision_history import DecisionHistory
from model_monitor.monitoring.types import MetricRecord
from model_monitor.storage.metrics_store import MetricsStore


MODEL_PATH = Path("models/current.pkl")
ACTIVE_FILE = Path("models/active.json")


class Predictor:
    """
    Stateful batch inference wrapper.

    Responsibilities:
    - Load and hot-reload models
    - Run inference
    - Compute batch metrics
    - Detect drift
    - Produce decisions

    Does NOT:
    - Execute retraining
    - Promote or rollback models
    - Aggregate metrics
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        model_path: Path = MODEL_PATH,
        active_file: Path = ACTIVE_FILE,
        reference_features: Optional[np.ndarray] = None,
        f1_baseline: Optional[float] = None,
    ) -> None:
        self.cfg = config
        self.model_path = model_path
        self.active_file = active_file

        self.model: Optional[Any] = None
        self._loaded_version: Optional[str] = None
        self._loaded_mtime: float | None = None
        # Tracks whether a model was already active when the predictor started.
        # Used to distinguish "initial silent load" from "model appeared later".
        self._model_existed_at_startup = self.model_path.exists()


        # Optional baseline (rollback logic must not fire without it)
        self.f1_baseline: Optional[float] = (
            float(f1_baseline) if f1_baseline is not None else None
        )

        self.batch_index = 0
        self.feature_names: list[str] = []

        # Monitoring & decisioning
        self.metrics_store = MetricsStore()
        self.decision_history = DecisionHistory()
        self.decision_engine = DecisionEngine(config=self.cfg)

        self.drift_monitor: Optional[DriftMonitor] = None
        if reference_features is not None:
            self.drift_monitor = DriftMonitor(
                reference_features=reference_features,
                config=self.cfg.drift,
            )

    # --------------------------------------------------
    # Reload logic
    # --------------------------------------------------
    def reload(self) -> None:
        if not self.model_path.exists():
            self.model = None
            self._loaded_version = None
            return

        self.model = joblib.load(self.model_path)
        self._loaded_version = self._load_active_version()
        self._loaded_mtime = self.model_path.stat().st_mtime


    # NOTE:
    # Initial model load is NOT considered a reload.
    # Only version changes trigger reload=True.
    def reload_if_changed(self) -> bool:
        """
        Reload model if (and only if) the active model changed.

        A change is detected if:
        - the active version changes, OR
        - the model file is replaced on disk
        """
        if not self.model_path.exists():
            return False

        current_version = self._load_active_version()
        current_mtime = self.model_path.stat().st_mtime

        # First-ever load
        if self._loaded_version is None:
            self.reload()

            # Reload only if the model appeared AFTER startup
            return not self._model_existed_at_startup


        version_changed = current_version != self._loaded_version
        file_changed = (
            self._loaded_mtime is not None
            and current_mtime != self._loaded_mtime
        )

        if version_changed or file_changed:
            self.reload()
            return True

        return False

    def _load_active_version(self) -> Optional[str]:
        if not self.active_file.exists():
            return None
        with self.active_file.open() as f:
            payload = json.load(f)
        return payload.get("version")

    @property
    def active_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("No active model loaded")
        return self.model

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def predict_batch(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        *,
        batch_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, Decision]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if not self.feature_names:
            self.feature_names = list(X.columns)

        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        X = X[self.feature_names]

        self.batch_index += 1
        start_ts = time.time()

        probs = self.active_model.predict_proba(X)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)

        accuracy = 0.0
        f1 = 0.0
        drift_score = 0.0

        if y_true is not None:
            accuracy = float(accuracy_score(y_true, preds))
            f1 = float(f1_score(y_true, preds, zero_division=0))

        if self.drift_monitor is not None:
            drift_score = float(self.drift_monitor.update(X.values))

        # ----------------------------------------------
        # Decision logic (TYPE-SAFE)
        # ----------------------------------------------
        baseline = self.f1_baseline  # for pylance narrowing
        has_labels = y_true is not None
        enough_samples = len(X) >= self.cfg.retrain.min_samples

        # Simple trust heuristic
        decision_trust_score = float(
            max(0.0, min(1.0, (1.0 - drift_score)))
        )

        if has_labels and baseline is not None and enough_samples:
            decision = self.decision_engine.decide(
                batch_index=self.batch_index,
                trust_score=decision_trust_score,
                f1=f1,
                f1_baseline=baseline,
                drift_score=drift_score,
                recent_actions=self.decision_history.recent_actions(),
            )
        else:
            decision = Decision(
                action="none",
                reason="insufficient_signal_for_decision",
                metadata={
                    "has_labels": has_labels,
                    "has_baseline": baseline is not None,
                    "n_samples": len(X),
                },
            )

        record: MetricRecord = {
            "timestamp": start_ts,
            "batch_id": batch_id,
            "n_samples": len(X),
            "accuracy": accuracy,
            "f1": f1,
            "avg_confidence": float(np.mean(confs)),
            "drift_score": drift_score,
            "decision_latency_ms": (time.time() - start_ts) * 1000.0,
            "action": decision.action,
            "reason": decision.reason,
            "previous_model": self._loaded_version,
            "new_model": None,
        }

        self.metrics_store.write(record)
        self.decision_history.record(decision)

        return preds, confs, decision

    def current_model_version(self) -> Optional[str]:
        return self._load_active_version()
