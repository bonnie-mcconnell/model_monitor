"""Predictor: stateful inference wrapper with drift monitoring and decisions."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from model_monitor.config.settings import AppConfig
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decision_history import DecisionHistory
from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.monitoring.drift import DriftMonitor
from model_monitor.monitoring.trust_score import compute_trust_score
from model_monitor.monitoring.types import MetricRecord
from model_monitor.utils.stats import expected_calibration_error

if TYPE_CHECKING:
    from model_monitor.monitoring.causal_drift import CausalDriftAttributor
    from model_monitor.monitoring.conformal import ConformalMonitor
    from model_monitor.monitoring.data_quality import DataQualityMonitor
    from model_monitor.monitoring.output_drift import OutputDriftMonitor
    from model_monitor.monitoring.raw_data_buffer import RawDataBuffer
    from model_monitor.monitoring.shap_attribution import ShapDriftAttributor
    from model_monitor.monitoring.threshold_advisor import ThresholdAdvisor

# Behavioral contracts (DecisionContext, DecisionRecord, DecisionOutcome) not available on main branch.

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/current.pkl")
ACTIVE_FILE = Path("models/active.json")

# Default per-request latency budget for behavioral evaluation.
# Evaluations that exceed this are skipped and logged so they never
# block inference - the monitoring pipeline picks up the signal
# asynchronously through BehavioralDecisionStore.violation_rate().
_DEFAULT_BEHAVIORAL_BUDGET_MS: float = 50.0


class Predictor:
    """
    Stateful batch inference wrapper.

    Responsibilities:
    - Load and hot-reload models
    - Run inference on each batch
    - Compute accuracy, F1, drift, latency, and trust score
    - Produce operational decisions via DecisionEngine
    - Optionally evaluate outputs against behavioral contracts

    Explicitly does NOT:
    - Persist metrics to any store
    - Execute model lifecycle actions (promote / retrain / rollback)

    Behavioral contract integration:
        Pass a ``BehavioralContractRunner`` at construction time to evaluate
        each batch output against versioned behavioral guarantees.  The runner
        is called with a per-request latency budget (``behavioral_budget_ms``);
        evaluations that exceed the budget are skipped and logged rather than
        blocking inference.  The resulting ``DecisionRecord`` is available on
        ``last_behavioral_record`` for the caller to persist or inspect.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        model_path: Path = MODEL_PATH,
        active_file: Path = ACTIVE_FILE,
        reference_features: np.ndarray | None = None,
        f1_baseline: float | None = None,
        behavioral_runner: None = None,
        behavioral_budget_ms: float = _DEFAULT_BEHAVIORAL_BUDGET_MS,
        stored_bin_edges: dict[int, np.ndarray] | None = None,
        raw_data_buffer: RawDataBuffer | None = None,
        shap_attributor: ShapDriftAttributor | None = None,
        output_drift_monitor: OutputDriftMonitor | None = None,
        data_quality_monitor: DataQualityMonitor | None = None,
        conformal_monitor: ConformalMonitor | None = None,
        causal_drift_attributor: CausalDriftAttributor | None = None,
        threshold_advisor: ThresholdAdvisor | None = None,
        regression_monitor: object | None = None,
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

        self._behavioral_runner = behavioral_runner
        self._behavioral_budget_ms = behavioral_budget_ms
        self.last_behavioral_record: None = None  # BM branch only
        # Exponential moving average of per-batch behavioral violation scores.
        # Updated by _run_behavioral_evaluation after each contract evaluation.
        # Passed to compute_trust_score so the behavioral signal is live in the
        # per-batch trust score, not only in the async aggregation loop.
        self._behavioral_ema: float = 0.0

        # Optional injectable components for retraining and SHAP attribution.
        self._raw_data_buffer = raw_data_buffer
        self._shap_attributor = shap_attributor
        # Optional additional monitors - all None by default.
        self._output_drift_monitor = output_drift_monitor
        self._data_quality_monitor = data_quality_monitor
        self._conformal_monitor = conformal_monitor
        self._causal_drift_attributor = causal_drift_attributor
        self._threshold_advisor = threshold_advisor
        # Regression monitor - optional; exposes regression metrics via the API.
        # Accepts any object with .history and .summary() (RegressionMonitor).
        self._regression_monitor = regression_monitor

        self.drift_monitor: DriftMonitor | None = None
        if reference_features is not None:
            self.drift_monitor = DriftMonitor(
                reference_features=reference_features,
                config=self.cfg.drift,
                stored_bin_edges=stored_bin_edges,
            )

        # Last-batch observability attributes - set on every predict_batch call
        # so callers never have to parse decision.metadata.
        self.last_drift_score: float = 0.0
        self.last_trust_score: float = 1.0
        self.last_behavioral_violation_rate: float = 0.0
        self.last_metric_record: MetricRecord | None = None

    # --------------------------------------------------
    # Reload logic
    # --------------------------------------------------
    def reload(self) -> None:
        """Reload the model from disk and reset per-model state.

        When the loaded model has a different feature set than the previous
        one, the ``RawDataBuffer`` is reset via ``reset_schema()`` so stale
        rows from the old schema are not used in the next retrain.
        """
        if not self.model_path.exists():
            self.model = None
            self._loaded_version = None
            self._loaded_mtime = None
            return

        prev_feature_names = list(self.feature_names)

        self.model = joblib.load(self.model_path)
        self._loaded_version = self._load_active_version()
        self._loaded_mtime = self.model_path.stat().st_mtime

        new_feature_names: list[str] | None = None
        if hasattr(self.model, "feature_names_in_"):
            new_feature_names = list(self.model.feature_names_in_)

        if (
            new_feature_names is not None
            and new_feature_names != prev_feature_names
            and self._raw_data_buffer is not None
        ):
            self._raw_data_buffer.reset_schema(new_feature_names)
            self.feature_names = new_feature_names

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
        behavioral_output: str | None = None,
        candidate_exists: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, Decision]:
        """
        Run inference on one batch and produce an operational decision.

        Args:
            X: feature matrix for this batch.
            y_true: ground-truth labels. When provided, enables accuracy/F1
                computation and unlocks the decision engine. When absent, the
                engine is skipped and action=DecisionType.NONE is returned.
            batch_id: caller-assigned identifier for this batch, included in
                decision metadata for end-to-end audit trail traceability.
            behavioral_output: raw model output string to evaluate against the
                behavioral contract (e.g. the JSON response for a generative
                model). When provided and a ``BehavioralContractRunner`` is
                configured, evaluation runs within ``behavioral_budget_ms``.
                Skipped when ``None`` or when no runner is configured.
            candidate_exists: whether a staged candidate model is available for
                promotion. Passed to the decision engine so the ``promote`` rule
                only fires when there is actually something to promote.

        Returns:
            (predictions, confidences, decision) - predictions and confidences
            always reflect the full batch; decision reflects current system state.
            ``last_behavioral_record`` is updated if evaluation ran.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if not self.feature_names:
            self.feature_names = list(X.columns)

        X = X[self.feature_names]
        self.batch_index += 1
        t_start = time.monotonic()

        # Batch inference - vectorised, as in production.
        # We time the full batch call once, then estimate per-sample tail latency
        # via a separate microbenchmark on a small subsample of ≤20 rows.
        # Running N single-row predict_proba calls in a Python loop would be ~100x
        # slower than batch inference and would produce meaningless timing numbers
        # (measuring sklearn dispatch overhead rather than real inference latency).
        import time as _time_mod

        n_samples_batch = len(X)

        probs = self.active_model.predict_proba(X)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        avg_confidence = float(confs.mean())

        # p95/p99 tail latency via single-row microbenchmark.
        # We time _TIMING_SUBSAMPLE_SIZE individual single-row calls on a random
        # subsample.  This approximates per-request latency in a real serving
        # stack (where each request is typically one row) while keeping total
        # overhead proportional to the subsample size, not the full batch.
        # Only run when the batch is large enough to give stable percentiles.
        _MIN_SAMPLES_FOR_PERCENTILES = 20
        _TIMING_SUBSAMPLE_SIZE = min(20, n_samples_batch)
        p95_latency_ms: float | None = None
        p99_latency_ms: float | None = None
        if n_samples_batch >= _MIN_SAMPLES_FOR_PERCENTILES:
            rng_idx = np.random.default_rng(self.batch_index)
            subsample_idx = rng_idx.integers(
                0, n_samples_batch, size=_TIMING_SUBSAMPLE_SIZE
            )
            sample_times: list[float] = []
            for idx in subsample_idx:
                _t0 = _time_mod.monotonic()
                self.active_model.predict_proba(X.iloc[[int(idx)]])
                sample_times.append((_time_mod.monotonic() - _t0) * 1_000)
            times_arr = np.array(sample_times)
            p95_latency_ms = float(np.percentile(times_arr, 95))
            p99_latency_ms = float(np.percentile(times_arr, 99))

        drift_score = (
            float(self.drift_monitor.update(X.values)) if self.drift_monitor else 0.0
        )
        feature_drift_scores: list[float] | None = (
            list(self.drift_monitor.last_feature_scores)
            if self.drift_monitor and self.drift_monitor.last_feature_scores
            else None
        )

        # Output drift: PSI on predicted probability distributions.
        output_drift_score: float | None = None
        output_drift_class_scores: list[float] | None = None
        if self._output_drift_monitor is not None:
            output_drift_score = float(self._output_drift_monitor.update(probs))
            if self._output_drift_monitor.last_class_scores:
                output_drift_class_scores = list(
                    self._output_drift_monitor.last_class_scores
                )

        # Data quality: check for nulls, out-of-range values, schema changes.
        data_quality_score: float | None = None
        if self._data_quality_monitor is not None:
            dq_report = self._data_quality_monitor.evaluate(X)
            data_quality_score = dq_report.quality_score

        # Causal drift attribution: when drift is detected, classify each
        # drifting feature as genuine_shift, pipeline_suspect, or correlated_follower.
        # Only runs when drift exceeds psi_threshold to avoid overhead on healthy batches.
        causal_drift_report: dict[str, object] | None = None
        if (
            self._causal_drift_attributor is not None
            and self.drift_monitor is not None
            and self.drift_monitor.last_feature_scores
            and drift_score >= self.cfg.drift.psi_threshold
        ):
            try:
                report = self._causal_drift_attributor.attribute(
                    X.values,
                    list(self.drift_monitor.last_feature_scores),
                )
                causal_drift_report = {
                    "dominant_cause": report.dominant_cause,
                    "recommendation": report.recommendation,
                    "n_suspects": report.n_suspects,
                    "n_genuine": report.n_genuine,
                    "features": [
                        {
                            "name": r.feature_name,
                            "psi": round(r.psi, 4),
                            "class": r.drift_class,
                            "explanation": r.explanation,
                        }
                        for r in report.feature_results
                        if r.drift_class != "stable"
                    ],
                }
            except Exception:
                pass  # causal attribution failure never blocks inference

        # Accumulate labeled data for retraining (API-driven path).
        if y_true is not None and self._raw_data_buffer is not None:
            try:
                self._raw_data_buffer.add_batch(
                    X.values,
                    np.asarray(y_true),
                    self.feature_names,
                )
            except (ValueError, TypeError):
                # Schema mismatch between current feature set and buffer.
                # Expected mid-flight after a model retrain with new features.
                pass

        # Compute SHAP importance shift when an attributor is configured.
        # Only triggered when drift is above the negligible threshold.
        shap_attribution: dict[str, float] | None = None
        if self._shap_attributor is not None and drift_score > 0.1:
            try:
                shap_attribution = self._shap_attributor.attribute(X.values)
            except (RuntimeError, ValueError, MemoryError):
                # SHAP can raise on unexpected input shapes or OOM.
                # Attribution is best-effort - never block inference.
                pass

        # Always update observability attributes so callers can read the
        # current drift level regardless of which decision branch fires.
        self.last_drift_score = drift_score

        # Run behavioral evaluation within budget before committing latency measurement.
        # This keeps the latency signal honest: behavioral overhead is included in the
        # decision_latency_ms fed to the trust score.
        if behavioral_output is not None and self._behavioral_runner is not None:
            self._run_behavioral_evaluation(
                output=behavioral_output,
                batch_id=batch_id,
            )

        # Expose the current EMA as a public attribute so callers and dashboards
        # can read the smoothed behavioral signal without digging into _behavioral_ema.
        self.last_behavioral_violation_rate = self._behavioral_ema

        # Narrow f1_baseline to float inside the condition so no assert is needed.
        # self.f1_baseline is float | None; the `is not None` check below narrows it.
        f1_baseline = self.f1_baseline

        if y_true is not None and f1_baseline is not None and self.batch_index > 1:
            accuracy = float(accuracy_score(y_true, preds))
            f1 = float(f1_score(y_true, preds, zero_division=0))
            latency_ms = (time.monotonic() - t_start) * 1_000

            correct = (preds == np.asarray(y_true)).astype(float)
            ece = expected_calibration_error(confs, correct)

            # behavioral_violation_rate feeds the behavioral trust component
            # directly at batch evaluation time.  The same signal also enters
            # the async aggregation loop via BehavioralDecisionStore - the two
            # paths are complementary: per-batch for fast response, aggregate
            # for stability across longer windows.
            # Conformal coverage monitoring - requires labeled data.
            conformal_coverage: float | None = None
            conformal_set_size: float | None = None
            if self._conformal_monitor is not None:
                try:
                    _conf_result = self._conformal_monitor.monitor(
                        probs, np.asarray(y_true)
                    )
                    conformal_coverage = _conf_result.coverage_rate
                    conformal_set_size = _conf_result.mean_set_size
                except Exception:
                    pass  # conformal monitor failure never blocks inference

            trust_score, _ = compute_trust_score(
                accuracy=accuracy,
                f1=f1,
                avg_confidence=avg_confidence,
                drift_score=drift_score,
                decision_latency_ms=latency_ms,
                calibration_error=ece,
                p95_latency_ms=p95_latency_ms,
                output_drift_score=output_drift_score,
                data_quality_score=data_quality_score,
                behavioral_violation_rate=self._behavioral_ema,
                config=self.cfg.trust_score,
            )

            # Adaptive threshold advisor: record stable-period signals only.
            # We guard on drift_score < psi_threshold - feeding drifted batches
            # corrupts the advisor's reference distribution and produces absurdly
            # high threshold recommendations (0.5–3.8 instead of ~0.01).
            if (
                self._threshold_advisor is not None
                and feature_drift_scores is not None
                and drift_score < self.cfg.drift.psi_threshold
            ):
                try:
                    self._threshold_advisor.observe(
                        psi_scores=feature_drift_scores,
                        trust_score=float(trust_score),
                    )
                except Exception:
                    pass  # advisor errors never block inference

            self.last_trust_score = trust_score

            decision = self.decision_engine.decide(
                batch_index=self.batch_index,
                trust_score=trust_score,
                f1=f1,
                f1_baseline=f1_baseline,
                drift_score=drift_score,
                recent_actions=self.decision_history.recent_actions(),
                candidate_exists=candidate_exists,
            )
        else:
            accuracy = (
                float(accuracy_score(y_true, preds)) if y_true is not None else 0.0
            )
            f1 = (
                float(f1_score(y_true, preds, zero_division=0))
                if y_true is not None
                else 0.0
            )
            latency_ms = (time.monotonic() - t_start) * 1_000
            ece = None
            conformal_coverage = None
            conformal_set_size = None
            # Unlabeled conformal: track set size without coverage computation.
            if self._conformal_monitor is not None:
                try:
                    _conf_result = self._conformal_monitor.monitor(probs)
                    conformal_set_size = _conf_result.mean_set_size
                except Exception:
                    pass
            decision = Decision(
                action=DecisionType.NONE,
                reason="insufficient_signal_for_decision",
                metadata={"batch_id": batch_id, "n_samples": len(X)},
            )

        self.last_metric_record = {
            "timestamp": time.time(),
            "batch_id": batch_id,
            "n_samples": len(X),
            "accuracy": accuracy,
            "f1": f1,
            "avg_confidence": avg_confidence,
            "drift_score": drift_score,
            "feature_drift_scores": feature_drift_scores,
            "output_drift_score": output_drift_score,
            "output_drift_class_scores": output_drift_class_scores,
            "calibration_error": ece,
            "decision_latency_ms": latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "p99_latency_ms": p99_latency_ms,
            "data_quality_score": data_quality_score,
            "conformal_coverage": conformal_coverage if y_true is not None else None,
            "conformal_set_size": conformal_set_size,
            # Persist the current EMA so the behavioral signal is auditable
            # in the metrics store and queryable from dashboards/Prometheus.
            # None when no BehavioralContractRunner is configured.
            "behavioral_violation_rate": (
                self._behavioral_ema if self._behavioral_runner is not None else None
            ),
            "causal_drift_report": causal_drift_report,
            "mmd_p_value": None,
            "mmd_is_drift": None,
            "shap_attribution": shap_attribution,
            "action": decision.action,
            "reason": decision.reason,
            "previous_model": None,
            "new_model": None,
        }

        self.decision_history.record(decision)
        return preds, confs, decision

    def _run_behavioral_evaluation(
        self,
        *,
        output: str,
        batch_id: str,
    ) -> None:
        """No-op on main branch - behavioral contracts live on behavior-monitoring branch."""
        return

    def current_model_version(self) -> str | None:
        """
        Return the currently active model version as declared in active.json.
        Does not load or reload the model.
        """
        return self._load_active_version()
