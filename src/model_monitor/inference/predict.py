# src/model_monitor/inference/predict.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple, Any

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score  # type: ignore

from model_monitor.config.settings import AppConfig
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import Decision
from model_monitor.core.decision_history import DecisionHistory
from model_monitor.monitoring.drift import DriftMonitor


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

    Explicitly does NOT:
    - Persist metrics
    - Execute actions
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
        self._loaded_mtime: Optional[float] = None
        self._model_existed_at_startup = self.model_path.exists()

        self.f1_baseline = float(f1_baseline) if f1_baseline is not None else None
        self.batch_index = 0
        self.feature_names: list[str] = []

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
            self._loaded_mtime = None
            return

        self.model = joblib.load(self.model_path)
        self._loaded_version = self._load_active_version()
        self._loaded_mtime = self.model_path.stat().st_mtime

    def reload_if_changed(self) -> bool:
        if not self.model_path.exists():
            return False

        current_version = self._load_active_version()
        current_mtime = self.model_path.stat().st_mtime

        if self._loaded_version is None:
            self.reload()
            return not self._model_existed_at_startup

        if (
            current_version != self._loaded_version
            or current_mtime != self._loaded_mtime
        ):
            self.reload()
            return True

        return False

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

        X = X[self.feature_names]
        self.batch_index += 1
        start_ts = time.time()

        probs = self.active_model.predict_proba(X)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)

        accuracy = (
            float(accuracy_score(y_true, preds)) if y_true is not None else 0.0
        )
        f1 = float(f1_score(y_true, preds, zero_division=0)) if y_true is not None else 0.0

        drift_score = (
            float(self.drift_monitor.update(X.values))
            if self.drift_monitor
            else 0.0
        )

        trust_proxy = max(0.0, min(1.0, 1.0 - drift_score))

        can_decide = (
            y_true is not None
            and self.f1_baseline is not None
            and self.batch_index > 1
        )

        baseline = self.f1_baseline
        assert baseline is not None  # for type-checkers only

        decision = (
            self.decision_engine.decide(
                batch_index=self.batch_index,
                trust_score=trust_proxy,
                f1=f1,
                f1_baseline=baseline,
                drift_score=drift_score,
                recent_actions=self.decision_history.recent_actions(),
            )
            if can_decide
            else Decision(
                action="none",
                reason="insufficient_signal_for_decision",
                metadata={"n_samples": len(X)},
            )
        )


        self.decision_history.record(decision)
        return preds, confs, decision
    

    def current_model_version(self) -> Optional[str]:
        """
        Return the currently active model version as declared in active.json.
        Does not load or reload the model.
        """
        return self._load_active_version()

