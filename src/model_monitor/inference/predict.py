"""Predictor: stateful inference wrapper with drift monitoring and decisions."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from model_monitor.config.settings import AppConfig
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decision_history import DecisionHistory
from model_monitor.core.decisions import Decision
from model_monitor.monitoring.drift import DriftMonitor
from model_monitor.monitoring.trust_score import compute_trust_score

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
        reference_features: np.ndarray | None = None,
        f1_baseline: float | None = None,
    ) -> None:
        self.cfg = config
        self.model_path = model_path
        self.active_file = active_file

        self.model: Any | None = None
        self._loaded_version: str | None = None
        self._loaded_mtime: float | None = None
        self._model_existed_at_startup = self.model_path.exists()

        self.f1_baseline = float(f1_baseline) if f1_baseline is not None else None
        self.batch_index = 0
        self.feature_names: list[str] = []

        self.decision_history = DecisionHistory()
        self.decision_engine = DecisionEngine(config=self.cfg)

        self.drift_monitor: DriftMonitor | None = None
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

    def _load_active_version(self) -> str | None:
        if not self.active_file.exists():
            return None
        with self.active_file.open() as f:
            raw = json.load(f).get("version")
            return str(raw) if raw is not None else None

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
        y_true: pd.Series | None = None,
        *,
        batch_id: str,
    ) -> tuple[np.ndarray, np.ndarray, Decision]:
        """
        Run inference on one batch and produce an operational decision.

        Args:
            X: feature matrix for this batch.
            y_true: ground-truth labels. When provided, enables accuracy/F1
                computation and unlocks the decision engine. When absent, the
                engine is skipped and action="none" is returned.
            batch_id: caller-assigned identifier for this batch, included in
                decision metadata for end-to-end audit trail traceability.

        Returns:
            (predictions, confidences, decision) — predictions and confidences
            always reflect the full batch; decision reflects current system state.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if not self.feature_names:
            self.feature_names = list(X.columns)

        X = X[self.feature_names]
        self.batch_index += 1
        t_start = time.monotonic()

        probs = self.active_model.predict_proba(X)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        avg_confidence = float(confs.mean())

        drift_score = (
            float(self.drift_monitor.update(X.values))
            if self.drift_monitor
            else 0.0
        )

        # Narrow f1_baseline to float inside the condition so no assert is needed.
        # self.f1_baseline is float | None; the `is not None` check below narrows it.
        f1_baseline = self.f1_baseline

        if y_true is not None and f1_baseline is not None and self.batch_index > 1:
            accuracy = float(accuracy_score(y_true, preds))
            f1 = float(f1_score(y_true, preds, zero_division=0))
            latency_ms = (time.monotonic() - t_start) * 1_000

            trust_score, _ = compute_trust_score(
                accuracy=accuracy,
                f1=f1,
                avg_confidence=avg_confidence,
                drift_score=drift_score,
                decision_latency_ms=latency_ms,
            )

            decision = self.decision_engine.decide(
                batch_index=self.batch_index,
                trust_score=trust_score,
                f1=f1,
                f1_baseline=f1_baseline,
                drift_score=drift_score,
                recent_actions=self.decision_history.recent_actions(),
            )
        else:
            decision = Decision(
                action="none",
                reason="insufficient_signal_for_decision",
                metadata={"batch_id": batch_id, "n_samples": len(X)},
            )

        self.decision_history.record(decision)
        return preds, confs, decision


    def current_model_version(self) -> str | None:
        """
        Return the currently active model version as declared in active.json.
        Does not load or reload the model.
        """
        return self._load_active_version()

