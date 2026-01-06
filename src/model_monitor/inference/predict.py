from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from model_monitor.config.settings import AppConfig
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import Decision
from model_monitor.monitoring.drift import DriftMonitor
from model_monitor.monitoring.decision_history import DecisionHistory
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.model_store import get_active_version


MODEL_PATH = Path("models/current.pkl")
SCHEMA_PATH = Path("data/reference/feature_schema.json")
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
    - Persist raw prediction logs
    - Execute retraining
    - Promote or rollback models
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        model_path: Path = MODEL_PATH,
        active_file: Path = ACTIVE_FILE,
        reference_features: Optional[np.ndarray] = None,
        f1_baseline: float = 0.85,
    ) -> None:
        self.cfg = config
        self.model_path = model_path
        self.active_file = active_file

        self.model: Optional[Any] = None
        self._loaded_version: Optional[str] = None

        self.f1_baseline = float(f1_baseline)
        self.batch_index = 0

        # -------------------------------
        # Feature schema
        # -------------------------------
        if SCHEMA_PATH.exists():
            with SCHEMA_PATH.open() as f:
                self.feature_names: list[str] = json.load(f)
        else:
            self.feature_names = []

        # -------------------------------
        # Monitoring & decisioning
        # -------------------------------
        self.metrics = MetricsStore()
        self.decision_history = DecisionHistory()
        self.decision_engine = DecisionEngine(config=self.cfg)

        self.drift_monitor: Optional[DriftMonitor] = None
        if reference_features is not None:
            self.drift_monitor = DriftMonitor(
                reference_features=reference_features,
                config=self.cfg.drift,
            )

        if self.model_path.exists():
            self.reload()

    # --------------------------------------------------
    # Version helpers
    # --------------------------------------------------
    def _load_active_version(self) -> Optional[str]:
        if not self.active_file.exists():
            return None
        with self.active_file.open() as f:
            return json.load(f).get("version")

    @property
    def active_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("No active model loaded")
        return self.model

    # --------------------------------------------------
    # Reload logic
    # --------------------------------------------------
    def reload(self) -> None:
        self.model = joblib.load(self.model_path)
        self._loaded_version = self._load_active_version()

    def reload_if_changed(self) -> bool:
        current_version = self._load_active_version()

        if self.model is None:
            if not self.model_path.exists():
                raise RuntimeError("Model file missing")
            self.reload()
            return True

        if current_version != self._loaded_version:
            self.reload()
            return True

        return False

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

        X = X.reindex(columns=self.feature_names)
        if X.isnull().any().any():
            raise ValueError("Input batch missing required features")

        self.batch_index += 1

        model = self.active_model
        probs = model.predict_proba(X)
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

        decision = self.decision_engine.decide(
            batch_index=self.batch_index,
            f1=f1,
            f1_baseline=self.f1_baseline,
            drift_score=drift_score,
        )

        model_version = get_active_version()

        # Persist batch metrics
        self.metrics.write(
            batch_id=batch_id,
            n_samples=len(X),
            accuracy=accuracy,
            f1=f1,
            avg_confidence=float(np.mean(confs)),
            drift_score=drift_score,
            decision_latency_ms=0.0,  # measured upstream
            action=decision.action,
            reason=decision.reason,
            previous_model=None,
            new_model=None,
        )

        # Record decision history
        self.decision_history.write(
            batch_index=self.batch_index,
            action=decision.action,
            reason=decision.reason,
            f1=f1,
            f1_baseline=self.f1_baseline,
            drift_score=drift_score,
            model_version=model_version,
        )

        return preds, confs, decision
