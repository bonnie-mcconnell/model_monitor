"""model_monitor.monitor - public SDK entry point.

This module provides the :class:`Monitor` class, which wraps any
scikit-learn-compatible model (or any callable that behaves like one) with the
full model_monitor stack in five lines::

    from model_monitor import Monitor

    monitor = Monitor(model, reference_data=X_train, feature_names=feature_cols)
    result  = monitor.predict(X_batch, y_true=y_batch)

    print(result.trust_score, result.drift_score, result.is_joint_drifting)
    print(monitor.report())

Design goals
------------
* **Zero required config.**  Reasonable defaults are inferred from the data.
  Pass a :class:`MonitorConfig` to customise anything.
* **Bring your own model.**  Accepts any object with a ``predict_proba``
  method (sklearn, xgboost, lightgbm, …) *or* a bare ``predict`` callable
  that returns class labels.
* **Incremental.**  Call ``predict`` batch-by-batch.  History accumulates
  in-process; persistence is opt-in via ``db_path``.
* **Readable.**  ``summary()`` returns a plain dict.  ``report()`` returns
  a human-readable string.  Nothing requires Streamlit or FastAPI to be
  running.

The class deliberately does **not** wrap the full ``Predictor`` internal
class - that class is an internal implementation detail tuned for the
demo simulation.  ``Monitor`` is a clean public interface built on the same
components.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from model_monitor.config.settings import (
    DriftConfig,
)
from model_monitor.core.decisions import DecisionType
from model_monitor.monitoring.causal_drift import CausalDriftAttributor
from model_monitor.monitoring.conformal import ConformalMonitor
from model_monitor.monitoring.cusum import CUSUMDetector, CUSUMResult
from model_monitor.monitoring.data_quality import DataQualityMonitor
from model_monitor.monitoring.drift import DriftMonitor
from model_monitor.monitoring.mmd import MMDDriftDetector, MMDDriftResult
from model_monitor.monitoring.output_drift import OutputDriftMonitor
from model_monitor.monitoring.threshold_advisor import ThresholdAdvisor
from model_monitor.monitoring.trust_score import compute_trust_score
from model_monitor.monitoring.types import MetricRecord
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.training.model_card import ModelEvaluation, build_model_card

log = logging.getLogger(__name__)


@runtime_checkable
class _PredictProbaModel(Protocol):
    """Structural type: any model with ``predict_proba``."""

    def predict_proba(self, X: Any) -> np.ndarray: ...  # noqa: D102


@runtime_checkable
class _PredictModel(Protocol):
    """Structural type: any model with ``predict`` only."""

    def predict(self, X: Any) -> np.ndarray: ...  # noqa: D102


# ─────────────────────────────────────────────────────────────────────────────
# MonitorSummary  (returned from summary())
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MonitorSummary:
    """Typed summary of the monitoring state across all processed batches.

    Returned by :meth:`Monitor.summary`.  All fields are ``None`` before
    the first ``predict()`` call.

    Attributes:
        n_batches:               Total batches processed so far.
        mean_trust_score:        Rolling mean trust score across all batches.
        latest_trust_score:      Trust score from the most recent batch.
        mean_drift_score:        Rolling mean max-PSI drift score.
        latest_drift_score:      Drift score from the most recent batch.
        mmd_drift_rate:          Fraction of MMD-evaluated batches where
                                 ``mmd_is_drift`` was True.  ``None`` when
                                 MMD is disabled or no evaluations ran yet.
        latest_mmd_p_value:      MMD p-value from the most recent MMD batch.
        n_features:              Number of monitored features.
        feature_names:           Feature names in order.
        recommended_psi_warn_global:  Calibrated PSI warn threshold from the
                                 threshold advisor.  ``None`` until enough
                                 stable-period batches have been observed.
        recommended_trust_warn:  Calibrated trust score warn threshold.
        cusum_alarm_rate:        Fraction of CUSUM-evaluated batches where an
                                 alarm fired.  ``None`` when CUSUM is disabled.
    """

    n_batches: int = 0
    mean_trust_score: float | None = None
    latest_trust_score: float | None = None
    mean_drift_score: float | None = None
    latest_drift_score: float | None = None
    mmd_drift_rate: float | None = None
    latest_mmd_p_value: float | None = None
    n_features: int = 0
    feature_names: list[str] = dataclasses.field(default_factory=list)
    recommended_psi_warn_global: float | None = None
    recommended_trust_warn: float | None = None
    cusum_alarm_rate: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MonitorConfig:
    """All tunable parameters for a :class:`Monitor` instance.

    Every parameter has a sensible default.  You only need to supply the
    ones you want to override.

    Args:
        psi_threshold:      PSI value above which a feature is considered
                            drifting.  Industry default is 0.20 (severe) or
                            0.10 (warn).  Default: 0.10.
        drift_window:       Number of recent batches pooled to estimate the
                            current distribution for PSI.  Smaller = faster
                            detection; larger = fewer false positives.
                            Default: 5.
        conformal_alpha:    Miscoverage rate for conformal prediction sets.
                            0.10 → sets cover the true label 90% of the time.
                            Default: 0.10.
        trust_warn:         Trust score below which :attr:`BatchResult.is_healthy`
                            is ``False``.  Default: 0.70.
        trust_critical:     Trust score below which the system is considered in
                            a critical state.  Exposed on ``BatchResult`` for
                            callers to gate their own alerting logic.
                            Default: 0.50.
        enable_causal:      Wire in Granger-causal drift attribution.  Adds
                            ~10 ms per drifted batch; negligible otherwise.
                            Default: True.
        enable_conformal:   Calibrate a conformal monitor on reference data.
                            Requires ``y_reference`` labels.  Default: True
                            when labels are available.
        enable_threshold_advisor:
                            Record stable-period statistics so the advisor
                            can recommend calibrated thresholds after
                            ``min_batches`` observations.  Default: True.
        enable_mmd:         Enable the MMD joint distribution drift test.
                            Default: True.
        mmd_permutations:   Number of permutations for the MMD p-value.
                            Higher = more accurate at the cost of latency.
                            Default: 200.
        mmd_alpha:          Significance level for the MMD test.  Default: 0.05.
        mmd_every:          Run the MMD test every N batches.  1 = every batch
                            (most sensitive), 5 = every 5th batch (lower
                            latency overhead).  Default: 1.
        db_path:            Path to a SQLite database for persisting records.
                            ``None`` (default) keeps everything in memory.
        min_advisor_batches:
                            Minimum stable-period batches before the threshold
                            advisor emits recommendations.  Default: 20.
        cusum_delta:        CUSUM allowance - half the smallest PSI shift worth
                            detecting.  Rule of thumb: 0.5 × (target_shift − stable_psi).
                            0 (default) disables CUSUM.
        cusum_threshold:    CUSUM decision threshold h.  Alarm fires when S_t^+ > h.
                            Rule of thumb: 4 × stable_period_PSI_std.
                            0 (default) disables CUSUM.
        cusum_warmup:       Batches used to estimate the stable-period PSI mean
                            before CUSUM starts running.  Default: 5.
    """

    psi_threshold: float = 0.10
    drift_window: int = 5
    conformal_alpha: float = 0.10
    trust_warn: float = 0.70
    trust_critical: float = 0.50
    enable_causal: bool = True
    enable_conformal: bool = True
    enable_threshold_advisor: bool = True
    enable_mmd: bool = True
    mmd_permutations: int = 200
    mmd_alpha: float = 0.05
    mmd_every: int = 1
    db_path: str | None = None
    min_advisor_batches: int = 20
    # CUSUM sequential change-point detection.
    # When cusum_delta > 0 and cusum_threshold > 0, a CUSUMDetector is
    # constructed and updated on every batch's drift_score.
    # Set cusum_delta to half the smallest PSI shift worth detecting.
    # Set cusum_threshold to ~4× the stable-period PSI standard deviation.
    cusum_delta: float = 0.0  # 0 = disabled
    cusum_threshold: float = 0.0  # 0 = disabled
    cusum_warmup: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# BatchResult  (returned from predict())
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BatchResult:
    """Result of a single :meth:`Monitor.predict` call.

    Attributes:
        predictions:        Predicted class labels (integer array).
        confidences:        Max-class probability for each sample.
        trust_score:        Composite health score in [0, 1].
        drift_score:        Max per-feature PSI over the current window.
        psi_per_feature:    Per-feature PSI scores keyed by feature name.
                            ``None`` until the drift window has enough batches.
        batch_id:           Auto-generated UUID for this batch.
        causal_summary:     Dict with ``dominant_cause`` and ``recommendation``
                            when causal attribution ran; ``None`` otherwise.
        threshold_ready:    ``True`` once the advisor has enough stable-period
                            observations to recommend thresholds.
        mmd_result:         MMD two-sample test result; ``None`` when MMD is
                            disabled, fewer than 5 samples, or this batch was
                            not an MMD evaluation batch (see ``mmd_every``).

    Properties:
        is_healthy:         ``trust_score >= MonitorConfig.trust_warn``
        is_critical:        ``trust_score < MonitorConfig.trust_critical``
        is_drifting:        ``drift_score > MonitorConfig.psi_threshold``
        is_joint_drifting:  MMD test detected joint distribution shift.
    """

    predictions: np.ndarray
    confidences: np.ndarray
    trust_score: float
    drift_score: float
    batch_id: str
    psi_per_feature: dict[str, float] | None = None
    causal_summary: dict[str, str] | None = None
    threshold_ready: bool = False
    mmd_result: MMDDriftResult | None = None
    cusum_result: CUSUMResult | None = None
    # Config values threaded through from Monitor so properties work correctly.
    _trust_warn: float = 0.70
    _trust_critical: float = 0.50
    _psi_threshold: float = 0.10

    @property
    def is_healthy(self) -> bool:
        """True when trust_score is above the configured warn threshold."""
        return self.trust_score >= self._trust_warn

    @property
    def is_critical(self) -> bool:
        """True when trust_score is below the configured critical threshold."""
        return self.trust_score < self._trust_critical

    @property
    def is_drifting(self) -> bool:
        """True when max PSI exceeds the configured psi_threshold."""
        return self.drift_score > self._psi_threshold

    @property
    def is_joint_drifting(self) -> bool:
        """True when the MMD test detects joint distribution shift."""
        return self.mmd_result is not None and self.mmd_result.is_drift

    @property
    def is_cusum_alarm(self) -> bool:
        """True when CUSUM detects a change point in this batch.

        A CUSUM alarm fires at the *first* batch where accumulated evidence of a
        sustained shift exceeds the threshold - typically several batches before
        PSI would cross its own threshold.  ``False`` when CUSUM is not configured
        (set ``MonitorConfig.cusum_delta > 0`` to enable).
        """
        return self.cusum_result is not None and self.cusum_result.alarm


# ─────────────────────────────────────────────────────────────────────────────
# Monitor
# ─────────────────────────────────────────────────────────────────────────────


class Monitor:
    """Wrap any ML model with production-grade monitoring in five lines.

    Usage::

        from model_monitor import Monitor
        import numpy as np

        monitor = Monitor(clf, reference_data=X_train, feature_names=cols)
        result = monitor.predict(X_batch, y_true=y_batch)
        print(f"trust={result.trust_score:.3f}  drift={result.drift_score:.3f}")

    Parameters
    ----------
    model:
        Any object with ``predict_proba(X) -> ndarray`` *or* ``predict(X) ->
        ndarray``.  Accepts sklearn, XGBoost, LightGBM, CatBoost, or any
        callable wrapping an API model.
    reference_data:
        Training-time feature matrix used to establish the reference
        distribution for PSI drift detection.  Pass a numpy array or
        pandas DataFrame.  Required - there is no meaningful drift
        detection without a reference distribution.
    feature_names:
        Column names for ``reference_data``.  If ``reference_data`` is a
        DataFrame the names are taken from its columns and this argument
        is optional.
    y_reference:
        Training-time labels.  When provided, enables conformal calibration
        (20% held-out split) so :attr:`BatchResult.trust_score` includes
        the coverage component.
    config:
        :class:`MonitorConfig` instance.  Pass one to override any default.
    """

    def __init__(
        self,
        model: _PredictProbaModel | _PredictModel | Callable[..., np.ndarray],
        *,
        reference_data: np.ndarray | pd.DataFrame,
        feature_names: list[str] | None = None,
        y_reference: np.ndarray | pd.Series | None = None,
        config: MonitorConfig | None = None,
    ) -> None:
        self._model = model
        self._cfg = config or MonitorConfig()
        self._batch_count = 0
        self._history: list[dict[str, object]] = []

        # ── Resolve feature names ────────────────────────────────────────────
        if isinstance(reference_data, pd.DataFrame):
            self._feature_names: list[str] = (
                feature_names
                if feature_names is not None
                else list(reference_data.columns)
            )
            self._ref_np: np.ndarray = reference_data.values
        else:
            self._ref_np = np.asarray(reference_data)
            n_features = self._ref_np.shape[1]
            self._feature_names = (
                feature_names
                if feature_names is not None
                else [f"f{i}" for i in range(n_features)]
            )

        # ── Validate ─────────────────────────────────────────────────────────
        if len(self._feature_names) != self._ref_np.shape[1]:
            raise ValueError(
                f"feature_names has {len(self._feature_names)} entries but "
                f"reference_data has {self._ref_np.shape[1]} columns"
            )

        # ── Drift monitor ────────────────────────────────────────────────────

        _drift_cfg = DriftConfig(
            psi_threshold=self._cfg.psi_threshold,
            window=self._cfg.drift_window,
        )
        self._drift_monitor = DriftMonitor(
            reference_features=self._ref_np,
            config=_drift_cfg,
        )

        # ── Output drift monitor ─────────────────────────────────────────────
        # Subsample to at most 2000 rows - running predict_proba on the full
        # training set at init time is O(n_train) and can take several seconds
        # for large datasets.  2000 samples are sufficient to characterise the
        # reference output distribution for PSI binning.
        _ref_sample = self._ref_np
        if len(_ref_sample) > 2000:
            _rng = np.random.default_rng(0)
            _idx = _rng.choice(len(_ref_sample), 2000, replace=False)
            _ref_sample = _ref_sample[_idx]
        ref_probs = self._call_predict_proba(_ref_sample)
        self._output_drift: OutputDriftMonitor | None = None
        if ref_probs is not None:
            self._output_drift = OutputDriftMonitor(
                ref_probs,
                window=self._cfg.drift_window,
                threshold=self._cfg.psi_threshold,
            )

        # ── Data quality monitor ─────────────────────────────────────────────
        means = self._ref_np.mean(axis=0)
        stds = self._ref_np.std(axis=0) + 1e-9
        bounds = {
            name: (float(means[i] - 4 * stds[i]), float(means[i] + 4 * stds[i]))
            for i, name in enumerate(self._feature_names)
        }
        self._data_quality = DataQualityMonitor(
            feature_names=self._feature_names,
            feature_bounds=bounds,
            max_null_rate=0.05,
            max_oor_rate=0.02,
        )

        # ── Conformal monitor ────────────────────────────────────────────────
        self._conformal: ConformalMonitor | None = None
        y_ref_np: np.ndarray | None = None
        if y_reference is not None:
            y_ref_np = (
                np.asarray(y_reference)
                if not isinstance(y_reference, np.ndarray)
                else y_reference
            )
        if self._cfg.enable_conformal and y_ref_np is not None:
            # Use up to 20% of labeled reference rows for conformal calibration.
            # We re-run predict_proba only on the calibration slice to avoid
            # a second full-dataset inference pass.
            n_cal = max(50, len(y_ref_np) // 5)
            cal_probs = self._call_predict_proba(self._ref_np[:n_cal])
            if cal_probs is not None:
                self._conformal = ConformalMonitor(alpha=self._cfg.conformal_alpha)
                self._conformal.calibrate(
                    cal_probs,
                    y_ref_np[:n_cal].astype(int),
                )

        # ── Causal drift attributor ──────────────────────────────────────────
        self._causal: CausalDriftAttributor | None = None
        if self._cfg.enable_causal:
            self._causal = CausalDriftAttributor(
                feature_names=self._feature_names,
                psi_threshold=self._cfg.psi_threshold,
                alpha=0.05,
                max_lag=3,
            )
            self._causal.fit(self._ref_np)

        # ── Threshold advisor ────────────────────────────────────────────────
        self._advisor: ThresholdAdvisor | None = None
        if self._cfg.enable_threshold_advisor:
            self._advisor = ThresholdAdvisor(
                feature_names=self._feature_names,
                alpha=0.05,
                min_batches=self._cfg.min_advisor_batches,
            )

        # ── MMD multivariate drift detector ─────────────────────────────────
        # Detects joint distribution shift - the failure mode that PSI misses:
        # when correlations between features change but all marginals stay flat.
        self._mmd: MMDDriftDetector | None = None
        if self._cfg.enable_mmd:
            self._mmd = MMDDriftDetector(
                self._ref_np,
                alpha=self._cfg.mmd_alpha,
                n_permutations=self._cfg.mmd_permutations,
            )

        # ── CUSUM sequential change-point detector ───────────────────────────
        # Detects the exact batch where a sustained PSI shift begins.
        # Disabled when cusum_delta == 0 (the default).
        #
        # IMPORTANT: we do NOT hardcode reference_mean=0.0 because stable-period
        # PSI is never zero - it is typically 0.01–0.08 depending on the dataset.
        # Hardcoding reference_mean=0.0 causes every stable batch to accumulate
        # in s_pos and trigger false alarms within ~25 batches.
        #
        # Instead we defer construction until the first cusum_warmup batches have
        # been observed, then set reference_mean = mean(observed PSI) and build
        # the detector.  _cusum_warmup_psi accumulates observations until ready.
        self._cusum: CUSUMDetector | None = None
        self._cusum_warmup_psi: list[float] = []
        self._cusum_enabled: bool = (
            self._cfg.cusum_delta > 0 and self._cfg.cusum_threshold > 0
        )

        # ── Optional persistence ─────────────────────────────────────────────
        self._store: MetricsStore | None = None
        if self._cfg.db_path is not None:
            self._store = MetricsStore(db_path=self._cfg.db_path)

        # ── Streaming / per-request buffer ───────────────────────────────────
        # predict_one() accumulates individual rows here.  When the buffer
        # reaches _flush_every rows it is automatically flushed through the
        # full predict() pipeline and cleared.  This gives real-time inference
        # workloads access to all monitoring machinery (drift, CUSUM, MMD)
        # without requiring callers to manage batching manually.
        self._pending_rows: list[np.ndarray] = []
        self._pending_labels: list[int | float] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def warm_up(self, X: np.ndarray | pd.DataFrame) -> None:
        """Pre-populate the drift window with reference-distribution data.

        When a :class:`Monitor` is freshly constructed and ``drift_window=5``,
        the first ``drift_window - 1`` calls to :meth:`predict` return
        ``drift_score=0.0`` and ``psi_per_feature=None`` because the window
        hasn't filled yet.  This is misleading in a mid-deployment start: the
        first production batches appear artificially clean.

        Call ``warm_up()`` with representative stable-period data (e.g. a slice
        of the validation set) before the first production batch to pre-fill the
        window.  Warm-up data is **not** recorded in :attr:`history` and does
        not increment :attr:`n_batches`.

        Args:
            X: Feature matrix matching ``reference_data`` in shape and dtype.
               Typically a held-out slice of the training/validation set.

        Example::

            monitor = Monitor(clf, reference_data=X_train)
            monitor.warm_up(X_val[:200])          # fill the drift window
            result = monitor.predict(X_production) # PSI meaningful from batch 1
        """
        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        # Feed through the drift monitor window without recording a BatchResult.
        # Slice into drift_window-sized chunks so the buffer fills naturally;
        # if X is smaller than the window we feed it in one pass.
        chunk = max(1, len(X_np) // max(1, self._cfg.drift_window))
        for start in range(0, len(X_np), chunk):
            self._drift_monitor.update(X_np[start : start + chunk])

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        y_true: np.ndarray | pd.Series | None = None,
        *,
        batch_id: str | None = None,
    ) -> BatchResult:
        """Run one batch through the model and update all monitors.

        Parameters
        ----------
        X:
            Feature matrix for this batch.  Must have the same columns /
            features as ``reference_data``.
        y_true:
            Ground-truth labels for this batch.  When provided, F1 and
            conformal coverage are computed and included in the trust score.
        batch_id:
            Optional string identifier for this batch.  Auto-generated if
            not supplied.

        Returns
        -------
        :class:`BatchResult`
        """
        bid = batch_id or f"batch_{uuid.uuid4().hex[:8]}"
        self._batch_count += 1

        # Coerce to numpy
        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_np: np.ndarray | None = None
        if y_true is not None:
            y_np = (
                y_true.to_numpy()
                if isinstance(y_true, pd.Series)
                else np.asarray(y_true)
            )

        # ── Inference ───────────────────────────────────────────────────────
        t0 = time.monotonic()
        probs = self._call_predict_proba(X_np)
        if probs is not None:
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)
        else:
            preds = self._call_predict(X_np)
            # Synthesise uniform confidence when only labels are available
            confs = np.full(len(preds), 0.5)
            probs = None
        latency_ms = (time.monotonic() - t0) * 1000.0

        # ── Drift ────────────────────────────────────────────────────────────
        X_df = (
            pd.DataFrame(X_np, columns=self._feature_names)
            if not isinstance(X, pd.DataFrame)
            else X
        )
        drift_score = self._drift_monitor.update(X_np)
        psi_scores: list[float] = self._drift_monitor.last_feature_scores
        if not psi_scores:
            drift_score = 0.0

        # ── Output drift ─────────────────────────────────────────────────────
        output_drift: float | None = None
        if self._output_drift is not None and probs is not None:
            output_drift = self._output_drift.update(probs)

        # ── Data quality ─────────────────────────────────────────────────────
        dq_report = self._data_quality.evaluate(X_df)
        dq_score: float | None = (
            dq_report.quality_score if dq_report is not None else None
        )

        # ── Conformal coverage ───────────────────────────────────────────────
        conformal_cov: float | None = None
        if self._conformal is not None and probs is not None and y_np is not None:
            cm_result = self._conformal.monitor(probs, y_np.astype(int))
            conformal_cov = cm_result.coverage_rate

        # ── Accuracy / F1 ────────────────────────────────────────────────────
        accuracy: float | None = None
        f1: float | None = None
        if y_np is not None:
            accuracy = float(accuracy_score(y_np, preds))
            f1 = float(f1_score(y_np, preds, zero_division=0))

        # ── Trust score ──────────────────────────────────────────────────────
        trust_score_val, _ = compute_trust_score(
            accuracy=accuracy if accuracy is not None else 1.0,
            f1=f1 if f1 is not None else 1.0,
            avg_confidence=float(confs.mean()),
            drift_score=drift_score,
            decision_latency_ms=latency_ms,
            output_drift_score=output_drift,
            data_quality_score=dq_score,
        )
        trust_score = float(trust_score_val)

        # ── Per-feature PSI dict ─────────────────────────────────────────────
        psi_per_feature: dict[str, float] | None = (
            dict(zip(self._feature_names, psi_scores)) if psi_scores else None
        )

        # ── Causal attribution ───────────────────────────────────────────────
        causal_summary: dict[str, str] | None = None
        if self._causal is not None and drift_score > self._cfg.psi_threshold:
            try:
                report = self._causal.attribute(
                    X_np,
                    psi_scores=psi_scores,
                )
                causal_summary = {
                    "dominant_cause": str(report.dominant_cause),
                    "recommendation": str(report.recommendation),
                }
            except Exception:
                pass

        # ── Threshold advisor ────────────────────────────────────────────────
        advisor_ready = False
        if self._advisor is not None:
            if drift_score <= self._cfg.psi_threshold:
                # Only feed stable-period batches to the advisor.
                self._advisor.observe(
                    psi_scores=psi_scores
                    if psi_scores
                    else [0.0] * len(self._feature_names),
                    trust_score=trust_score,
                )
            advisor_ready = self._advisor.is_ready

        # ── MMD joint drift test ─────────────────────────────────────────────
        # MMD tests the full joint distribution - catches correlation-structure
        # changes that PSI never sees.  Run on every N batches (mmd_every) to
        # bound the per-batch latency cost; the permutation test is O(n²×B).
        mmd_result: MMDDriftResult | None = None
        self._batch_count_for_mmd = getattr(self, "_batch_count_for_mmd", 0) + 1
        if (
            self._mmd is not None
            and len(X_np) >= 5
            and self._batch_count_for_mmd % self._cfg.mmd_every == 0
        ):
            try:
                mmd_result = self._mmd.test(X_np)
            except Exception:
                pass

        # ── CUSUM change-point detection ─────────────────────────────────────
        # Auto-estimates reference_mean from the first cusum_warmup PSI values.
        # This calibrates the detector to this deployment's stable-period variance
        # rather than requiring the user to supply an arbitrary constant.
        cusum_result: CUSUMResult | None = None
        if self._cusum_enabled:
            if self._cusum is None:
                # Accumulate PSI observations until we have enough to estimate
                # the stable-period mean.
                self._cusum_warmup_psi.append(drift_score)
                n_needed = max(3, self._cfg.cusum_warmup)
                if len(self._cusum_warmup_psi) >= n_needed:
                    ref_mean = sum(self._cusum_warmup_psi) / len(self._cusum_warmup_psi)
                    self._cusum = CUSUMDetector(
                        reference_mean=ref_mean,
                        delta=self._cfg.cusum_delta,
                        threshold=self._cfg.cusum_threshold,
                        direction="up",
                        warmup_batches=0,  # already did warmup above
                    )
            else:
                try:
                    cusum_result = self._cusum.update(drift_score)
                except Exception:
                    pass

        # ── Record ───────────────────────────────────────────────────────────
        record: dict[str, object] = {
            "batch_id": bid,
            "timestamp": time.time(),
            "n_samples": len(X_np),
            "drift_score": drift_score,
            "trust_score": trust_score,
            "accuracy": accuracy,
            "f1": f1,
            "output_drift_score": output_drift,
            "data_quality_score": dq_score,
            "conformal_coverage": conformal_cov,
            "latency_ms": latency_ms,
            "causal_summary": causal_summary,
            "psi_per_feature": psi_per_feature,
            "mmd2": mmd_result.mmd2 if mmd_result is not None else None,
            "mmd_p_value": mmd_result.p_value if mmd_result is not None else None,
            "mmd_is_drift": mmd_result.is_drift if mmd_result is not None else None,
            "cusum_alarm": cusum_result.alarm if cusum_result is not None else None,
            "cusum_s_pos": cusum_result.s_pos if cusum_result is not None else None,
            "cusum_change_point": cusum_result.change_point
            if cusum_result is not None
            else None,
        }
        self._history.append(record)

        if self._store is not None:
            # Re-use already-computed accuracy/f1 - never recompute.
            store_rec: MetricRecord = {
                "batch_id": bid,
                "timestamp": time.time(),
                "n_samples": len(X_np),
                "accuracy": accuracy if accuracy is not None else 0.0,
                "f1": f1 if f1 is not None else 0.0,
                "avg_confidence": float(confs.mean()),
                "drift_score": drift_score,
                "decision_latency_ms": latency_ms,
                "p95_latency_ms": None,
                "p99_latency_ms": None,
                "calibration_error": None,
                "feature_drift_scores": list(psi_scores) if psi_scores else None,
                "output_drift_score": output_drift,
                "output_drift_class_scores": None,
                "data_quality_score": dq_score,
                "conformal_coverage": conformal_cov,
                "conformal_set_size": None,
                "behavioral_violation_rate": None,
                "causal_drift_report": None,
                "mmd_p_value": mmd_result.p_value if mmd_result is not None else None,
                "mmd_is_drift": mmd_result.is_drift if mmd_result is not None else None,
                "shap_attribution": None,
                "action": DecisionType.NONE,
                "reason": "sdk_batch",
                "previous_model": None,
                "new_model": None,
            }
            try:
                self._store.write(store_rec)
            except Exception as exc:
                log.warning("MetricsStore.write failed for batch %s: %s", bid, exc)

        result = BatchResult(
            predictions=preds,
            confidences=confs,
            trust_score=trust_score,
            drift_score=drift_score,
            batch_id=bid,
            psi_per_feature=psi_per_feature,
            causal_summary=causal_summary,
            threshold_ready=advisor_ready,
            mmd_result=mmd_result,
            cusum_result=cusum_result,
            _trust_warn=self._cfg.trust_warn,
            _trust_critical=self._cfg.trust_critical,
            _psi_threshold=self._cfg.psi_threshold,
        )

        # ── Fire alarm callbacks ─────────────────────────────────────────────
        callbacks = getattr(self, "_alarm_callbacks", [])
        for _cb, _fire_on in callbacks:
            if any(getattr(result, prop, False) for prop in _fire_on):
                try:
                    _cb(result)
                except Exception:
                    pass  # callbacks never block inference

        return result

    def predict_one(
        self,
        x: np.ndarray | pd.Series,
        y_true: int | float | None = None,
        *,
        flush_every: int = 64,
    ) -> np.ndarray:
        """Score a single sample and return the model's prediction.

        Real-time inference workloads - REST endpoints, streaming pipelines,
        per-event scoring - produce one row at a time.  ``predict_one`` is
        optimised for this pattern: the model's prediction is returned
        immediately, and monitoring (drift, CUSUM, MMD) runs once the buffer
        reaches ``flush_every`` rows rather than on every individual call.
        This avoids the overhead of running the full monitoring pipeline - PSI
        window update, conformal check, CUSUM update - on a batch of size 1,
        which would be both statistically meaningless and wasteful.

        The model's prediction for the current row is returned immediately,
        before the flush - monitoring always lags by at most ``flush_every``
        rows, which is the correct trade-off for latency-sensitive paths.

        Parameters
        ----------
        x:
            A single feature vector as a 1-D array or :class:`pandas.Series`.
            Must match the feature schema supplied at construction time.
        y_true:
            Ground-truth label for this sample, if available.  Accumulated
            labels are passed to :meth:`predict` at flush time so accuracy
            and F1 are computed over the flushed mini-batch.
        flush_every:
            Number of rows to accumulate before running the full monitoring
            pipeline.  Lower values increase monitoring frequency at the cost
            of more frequent (but cheaper) drift computations.  Default 64
            is a reasonable balance for sub-second inference latency.

        Returns
        -------
        numpy.ndarray
            The model's raw prediction for ``x`` - ``predict_proba`` output
            when the underlying model supports it, otherwise ``predict``.

        Example::

            monitor = Monitor(clf, reference_data=X_train, feature_names=cols)

            # In your request handler:
            for row in event_stream:
                pred = monitor.predict_one(row.features, flush_every=128)
                serve_prediction(pred)

            # Or flush remaining buffered rows explicitly at shutdown:
            monitor.flush()
        """
        x_np: np.ndarray = (
            np.asarray(x.values) if isinstance(x, pd.Series) else np.asarray(x)
        )
        if x_np.ndim != 1:
            raise ValueError(
                f"predict_one expects a 1-D feature vector; got shape {x_np.shape}. "
                "Use predict() for batches."
            )

        # Accumulate
        self._pending_rows.append(x_np)
        if y_true is not None:
            self._pending_labels.append(y_true)

        # Flush when the buffer is full
        if len(self._pending_rows) >= flush_every:
            self.flush()

        # Return the model's prediction for this row immediately.
        # We call the model directly rather than going through the full predict()
        # pipeline so the caller gets their result before the (deferred) flush.
        x_2d = x_np.reshape(1, -1)
        proba = self._call_predict_proba(x_2d)
        if proba is not None:
            return np.asarray(proba[0])
        return np.asarray(self._call_predict(x_2d))

    def flush(self) -> BatchResult | None:
        """Flush all buffered ``predict_one`` rows through the monitoring pipeline.

        Under normal operation the buffer is flushed automatically when it
        reaches ``flush_every`` rows.  Call ``flush()`` explicitly at process
        shutdown or at the end of a request burst to ensure no rows are left
        unmonitored.

        Returns
        -------
        :class:`BatchResult` or ``None``
            The monitoring result for the flushed batch, or ``None`` if the
            buffer was empty.
        """
        if not self._pending_rows:
            return None

        X_batch = np.stack(self._pending_rows, axis=0)
        y_batch: np.ndarray | None = (
            np.asarray(self._pending_labels) if self._pending_labels else None
        )
        self._pending_rows = []
        self._pending_labels = []
        return self.predict(X_batch, y_true=y_batch)

    def write_model_card(
        self,
        path: str | Path,
        *,
        model_version: int = 1,
        evaluation_f1: float | None = None,
        f1_improvement: float = 0.0,
        n_eval_samples: int | None = None,
        promotion_reason: str = "initial_deployment",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write a :class:`ModelCard` capturing training provenance to disk.

        Call once after constructing the monitor to record the model version,
        training data hash, feature schema, and evaluation metrics.  This is
        the SDK path for provenance recording when the internal retraining
        pipeline is not in use.

        Args:
            path:             Output path for the card JSON file.
            model_version:    Integer version number.  Increment on each retrain.
            evaluation_f1:    Held-out F1 score.  Defaults to the mean F1 from
                              monitoring history when available, otherwise 0.0.
            f1_improvement:   F1 improvement over the previous model version.
            n_eval_samples:   Evaluation set size.  Defaults to len(reference_data).
            promotion_reason: Human-readable deployment reason.
            extra:            Arbitrary metadata (hyperparameters, experiment IDs…).

        Example::

            monitor = Monitor(clf, reference_data=X_train, feature_names=cols)
            monitor.write_model_card(
                "cards/v1_card.json",
                model_version=1,
                evaluation_f1=val_f1,
                extra={"n_estimators": 100, "mlflow_run": "abc123"},
            )
        """
        if evaluation_f1 is None:
            f1_vals = [
                float(r["f1"])  # type: ignore[arg-type]
                for r in self._history
                if r.get("f1") is not None
            ]
            evaluation_f1 = float(sum(f1_vals) / len(f1_vals)) if f1_vals else 0.0

        card = build_model_card(
            model_version=model_version,
            X_train=self._ref_np,
            feature_names=self._feature_names,
            evaluation=ModelEvaluation(
                accuracy=evaluation_f1,
                f1=evaluation_f1,
                f1_improvement=f1_improvement,
                n_eval_samples=n_eval_samples or len(self._ref_np),
            ),
            promotion_reason=promotion_reason,
            extra=extra or {},
        )
        card.save(path)

    def on_alarm(
        self,
        callback: Callable[[BatchResult], None],
        *,
        fire_on: tuple[str, ...] = (
            "is_drifting",
            "is_joint_drifting",
            "is_cusum_alarm",
            "is_critical",
        ),
    ) -> None:
        """Register a callback that fires when an alarm condition is met.

        The callback receives the :class:`BatchResult` for the triggering batch.
        Multiple callbacks can be registered; they fire in registration order.
        This enables custom alerting (Slack, PagerDuty, email) without requiring
        the FastAPI stack.

        Args:
            callback:  Callable that accepts a :class:`BatchResult`.
            fire_on:   Tuple of :class:`BatchResult` property names to check.
                       The callback fires when *any* of them is True.
                       Default: ``("is_drifting", "is_joint_drifting",
                       "is_cusum_alarm", "is_critical")``.

        Example::

            def alert(result: BatchResult) -> None:
                print(f"ALARM batch={result.batch_id} trust={result.trust_score:.3f}")

            monitor.on_alarm(alert)
            monitor.on_alarm(
                lambda r: requests.post(SLACK_URL, json={"text": f"drift: {r.drift_score}"}),
                fire_on=("is_drifting",),
            )
        """
        if not hasattr(self, "_alarm_callbacks"):
            self._alarm_callbacks: list[
                tuple[Callable[[BatchResult], None], tuple[str, ...]]
            ] = []
        self._alarm_callbacks.append((callback, fire_on))

    def reset_after_retrain(self) -> None:
        """Reset monitoring state after a model retrain-and-promote event.

        Clears the drift window buffer, CUSUM cumulative sums, and the MMD
        batch counter so the post-retrain period is evaluated cleanly against
        the new model's reference distribution.  The history and batch count
        are *not* reset - they are cumulative across the deployment lifetime.

        Call this immediately after swapping in a new model to avoid:
          - PSI computed against the old model's drift window
          - CUSUM s_pos carrying over pre-retrain accumulated evidence
          - The first post-retrain batch appearing to have immediate drift

        Example::

            new_model = train_new_model(X_train, y_train)
            monitor._model = new_model
            monitor.reset_after_retrain()
            result = monitor.predict(X_first_post_retrain_batch)
        """
        # Reset drift window
        self._drift_monitor.buffer = deque(maxlen=self._cfg.drift_window)

        # Reset CUSUM - must re-estimate reference_mean from the new stable period
        if self._cusum_enabled:
            self._cusum = None
            self._cusum_warmup_psi = []

        # Reset MMD batch counter
        self._batch_count_for_mmd = 0

        # Discard any pending predict_one rows - they belong to the pre-retrain
        # distribution and would contaminate the first post-retrain monitoring window.
        self._pending_rows = []
        self._pending_labels = []

    def summary(self) -> MonitorSummary:
        """Return a typed summary of the monitoring state so far.

        Returns a :class:`MonitorSummary` with zero/None values before the
        first :meth:`predict` call.
        """
        if not self._history:
            return MonitorSummary(
                n_features=len(self._feature_names), feature_names=self._feature_names
            )

        trust_scores = [
            float(r["trust_score"])  # type: ignore[arg-type]
            for r in self._history
            if r.get("trust_score") is not None
        ]
        drift_scores = [
            float(r["drift_score"])  # type: ignore[arg-type]
            for r in self._history
            if r.get("drift_score") is not None
        ]

        # MMD drift rate: fraction of batches where MMD actually ran and detected drift.
        mmd_ran = [r for r in self._history if r.get("mmd_p_value") is not None]
        mmd_drift_rate: float | None = None
        latest_mmd_p: float | None = None
        if mmd_ran:
            mmd_drift_rate = sum(1 for r in mmd_ran if r.get("mmd_is_drift")) / len(
                mmd_ran
            )
            latest_mmd_p = float(mmd_ran[-1]["mmd_p_value"])  # type: ignore[arg-type]

        # CUSUM alarm rate
        cusum_batches = [r for r in self._history if r.get("cusum_alarm") is not None]
        cusum_alarm_rate: float | None = None
        if cusum_batches:
            cusum_alarm_rate = sum(
                1 for r in cusum_batches if r.get("cusum_alarm")
            ) / len(cusum_batches)

        s = MonitorSummary(
            n_batches=self._batch_count,
            mean_trust_score=float(np.mean(trust_scores)) if trust_scores else None,
            latest_trust_score=trust_scores[-1] if trust_scores else None,
            mean_drift_score=float(np.mean(drift_scores)) if drift_scores else None,
            latest_drift_score=drift_scores[-1] if drift_scores else None,
            mmd_drift_rate=mmd_drift_rate,
            latest_mmd_p_value=latest_mmd_p,
            n_features=len(self._feature_names),
            feature_names=list(self._feature_names),
            cusum_alarm_rate=cusum_alarm_rate,
        )

        if self._advisor is not None and self._advisor.is_ready:
            try:
                rec = self._advisor.recommend()
                s.recommended_psi_warn_global = rec.psi_warn_global
                s.recommended_trust_warn = rec.trust_warn
            except Exception:
                pass

        return s

    def report(self) -> str:
        """Return a human-readable monitoring report string."""
        s = self.summary()
        if s.n_batches == 0:
            return "Monitor - no batches processed yet."

        lines = [
            "─" * 52,
            f"  model_monitor - {s.n_batches} batch(es) processed",
            "─" * 52,
            (
                f"  Trust score  (mean) : {s.mean_trust_score:.4f}"
                if s.mean_trust_score is not None
                else "  Trust score  (mean) : n/a"
            ),
            (
                f"  Trust score (latest): {s.latest_trust_score:.4f}"
                if s.latest_trust_score is not None
                else "  Trust score (latest): n/a"
            ),
            (
                f"  Drift score  (mean) : {s.mean_drift_score:.4f}"
                if s.mean_drift_score is not None
                else "  Drift score  (mean) : n/a"
            ),
            (
                f"  Drift score (latest): {s.latest_drift_score:.4f}"
                if s.latest_drift_score is not None
                else "  Drift score (latest): n/a"
            ),
            f"  Features monitored  : {s.n_features}",
        ]

        if s.mmd_drift_rate is not None:
            lines.append(f"  MMD joint drift rate: {s.mmd_drift_rate:.2%}")
        if s.latest_mmd_p_value is not None:
            lines.append(f"  Latest MMD p-value  : {s.latest_mmd_p_value:.4f}")
        if s.recommended_psi_warn_global is not None:
            lines.append(f"  Recommended PSI warn: {s.recommended_psi_warn_global:.4f}")
        if s.recommended_trust_warn is not None:
            lines.append(f"  Recommended trust ↓ : {s.recommended_trust_warn:.4f}")

        lines.append("─" * 52)
        return "\n".join(lines)

    def threshold_recommendations(self) -> dict[str, float] | None:
        """Return calibrated threshold recommendations or ``None`` if not ready."""
        if self._advisor is None or not self._advisor.is_ready:
            return None
        rec = self._advisor.recommend()
        return {
            "psi_warn_global": rec.psi_warn_global,
            "trust_warn": rec.trust_warn,
            **{
                f"psi_{name}": v
                for name, v in zip(rec.feature_names, rec.psi_warn_per_feature)
            },
        }

    @property
    def history(self) -> list[dict[str, object]]:
        """Full list of per-batch monitoring records, newest-first."""
        return list(reversed(self._history))

    @property
    def n_batches(self) -> int:
        """Number of batches processed so far."""
        return self._batch_count

    def save(self, path: str | Path) -> None:
        """Persist the monitoring state to a JSON file.

        Saved state includes:
          - drift window buffers (for PSI continuity across restarts)
          - threshold advisor observations
          - MMD bandwidth (so it doesn't need to be re-estimated)
          - batch count and full in-memory history

        State is serialised as a self-contained JSON file.  The model weights
        and reference data are *not* saved - pass the same model and
        ``reference_data`` when calling :meth:`load`.

        Args:
            path: File path.  Parent directories are created if needed.

        Example::

            monitor.save("checkpoints/monitor_state.json")
            # ... process restarts ...
            monitor = Monitor.load("checkpoints/monitor_state.json", model=clf,
                                   reference_data=X_train)
        """
        import json as _json

        state: dict[str, object] = {
            "_version": 1,
            "_batch_count": self._batch_count,
            "_batch_count_for_mmd": getattr(self, "_batch_count_for_mmd", 0),
            "_feature_names": self._feature_names,
            "_history": self._history,
            # Pending predict_one rows are serialised so they survive a
            # checkpoint/restore cycle - e.g. a rolling restart mid-batch.
            "_pending_rows": [row.tolist() for row in self._pending_rows],
            "_pending_labels": list(self._pending_labels),
            "_cfg": {
                "psi_threshold": self._cfg.psi_threshold,
                "drift_window": self._cfg.drift_window,
                "conformal_alpha": self._cfg.conformal_alpha,
                "trust_warn": self._cfg.trust_warn,
                "trust_critical": self._cfg.trust_critical,
                "enable_causal": self._cfg.enable_causal,
                "enable_conformal": self._cfg.enable_conformal,
                "enable_threshold_advisor": self._cfg.enable_threshold_advisor,
                "enable_mmd": self._cfg.enable_mmd,
                "mmd_permutations": self._cfg.mmd_permutations,
                "mmd_alpha": self._cfg.mmd_alpha,
                "mmd_every": self._cfg.mmd_every,
                "db_path": self._cfg.db_path,
                "min_advisor_batches": self._cfg.min_advisor_batches,
            },
        }

        # Drift monitor: save window buffer (deque of numpy arrays)
        # DriftMonitor.buffer is a deque[np.ndarray]; we serialise each array
        # as a nested list.  The buffer is used for PSI computation - restoring
        # it means the first post-load batch uses real history, not a cold start.
        drift_buffer = list(self._drift_monitor.buffer)
        if drift_buffer:
            state["_drift_window"] = [arr.tolist() for arr in drift_buffer]

        # Threshold advisor: save raw observations
        if self._advisor is not None:
            state["_advisor_trust"] = list(self._advisor._trust_observations)
            state["_advisor_psi"] = [
                list(row) for row in self._advisor._psi_observations
            ]

        # MMD bandwidth
        if self._mmd is not None:
            state["_mmd_bandwidth"] = self._mmd.bandwidth

        # CUSUM state - cumulative sums and warmup buffer survive restarts.
        if self._cusum_enabled:
            state["_cusum_warmup_psi"] = list(self._cusum_warmup_psi)
            if self._cusum is not None:
                state["_cusum_s_pos"] = self._cusum.s_pos
                state["_cusum_s_neg"] = self._cusum.s_neg
                state["_cusum_n"] = self._cusum.n_batches
                state["_cusum_reference_mean"] = self._cusum.reference_mean

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        model: _PredictProbaModel | _PredictModel | Callable[..., np.ndarray],
        *,
        reference_data: np.ndarray | pd.DataFrame,
        feature_names: list[str] | None = None,
        y_reference: np.ndarray | pd.Series | None = None,
    ) -> Monitor:
        """Restore a :class:`Monitor` from a state file written by :meth:`save`.

        The ``model`` and ``reference_data`` arguments must be supplied again -
        they are not serialised.  Pass the same values that were used when the
        original monitor was constructed.

        Args:
            path:           Path to the state file.
            model:          The model to monitor (same as original).
            reference_data: Reference feature matrix (same as original).
            feature_names:  Column names (inferred from state file when absent).
            y_reference:    Reference labels for conformal calibration.

        Returns:
            A :class:`Monitor` with restored internal state.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            ValueError: if the state file is from an incompatible version.
        """
        import json as _json

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Monitor state file not found: {p}")

        state = _json.loads(p.read_text(encoding="utf-8"))
        if state.get("_version", 0) != 1:
            raise ValueError(
                f"Incompatible state file version {state.get('_version')}; expected 1"
            )

        cfg_data = state.get("_cfg", {})
        cfg = MonitorConfig(
            psi_threshold=cfg_data.get("psi_threshold", 0.10),
            drift_window=cfg_data.get("drift_window", 5),
            conformal_alpha=cfg_data.get("conformal_alpha", 0.10),
            trust_warn=cfg_data.get("trust_warn", 0.70),
            trust_critical=cfg_data.get("trust_critical", 0.50),
            enable_causal=cfg_data.get("enable_causal", True),
            enable_conformal=cfg_data.get("enable_conformal", True),
            enable_threshold_advisor=cfg_data.get("enable_threshold_advisor", True),
            enable_mmd=cfg_data.get("enable_mmd", True),
            mmd_permutations=cfg_data.get("mmd_permutations", 200),
            mmd_alpha=cfg_data.get("mmd_alpha", 0.05),
            mmd_every=cfg_data.get("mmd_every", 1),
            db_path=cfg_data.get("db_path"),
            min_advisor_batches=cfg_data.get("min_advisor_batches", 20),
        )

        # Reconstruct with original args so all components are initialised.
        names: list[str] | None = feature_names or state.get("_feature_names")
        m = cls(
            model,
            reference_data=reference_data,
            feature_names=names,
            y_reference=y_reference,
            config=cfg,
        )

        # Restore batch counters and history.
        m._batch_count = int(state.get("_batch_count", 0))
        m._batch_count_for_mmd = int(state.get("_batch_count_for_mmd", 0))
        history = state.get("_history")
        if isinstance(history, list):
            m._history = list(history)

        # Restore pending predict_one buffer so in-flight rows aren't lost
        # across checkpoint/restore cycles.
        pending_rows = state.get("_pending_rows")
        if isinstance(pending_rows, list):
            m._pending_rows = [np.asarray(r, dtype=float) for r in pending_rows]
        pending_labels = state.get("_pending_labels")
        if isinstance(pending_labels, list):
            m._pending_labels = list(pending_labels)

        # Restore drift window buffer so PSI is computed from real history,
        # not from an empty cold-start deque.
        drift_window = state.get("_drift_window")
        if isinstance(drift_window, list):
            restored_bufs = [np.asarray(arr, dtype=float) for arr in drift_window]
            m._drift_monitor.buffer = deque(
                restored_bufs, maxlen=m._drift_monitor.window
            )

        # Restore threshold advisor observations.
        if m._advisor is not None:
            trust_obs = state.get("_advisor_trust")
            psi_obs = state.get("_advisor_psi")
            if isinstance(trust_obs, list):
                m._advisor._trust_observations = list(trust_obs)
            if isinstance(psi_obs, list):
                m._advisor._psi_observations = [list(row) for row in psi_obs]

        # Restore MMD bandwidth so it's consistent with prior batches.
        if m._mmd is not None:
            bw = state.get("_mmd_bandwidth")
            if isinstance(bw, (int, float)):
                m._mmd._sigma = float(bw)

        # Restore CUSUM state for continuous change-point detection.
        warmup_psi = state.get("_cusum_warmup_psi")
        if isinstance(warmup_psi, list):
            m._cusum_warmup_psi = [float(v) for v in warmup_psi]
        ref_mean = state.get("_cusum_reference_mean")
        if m._cusum_enabled and isinstance(ref_mean, (int, float)):
            # Detector was already fitted before save - rebuild it.
            m._cusum = CUSUMDetector(
                reference_mean=float(ref_mean),
                delta=cfg.cusum_delta,
                threshold=cfg.cusum_threshold,
                direction="up",
                warmup_batches=0,
            )
            s_pos = state.get("_cusum_s_pos")
            s_neg = state.get("_cusum_s_neg")
            n = state.get("_cusum_n")
            if isinstance(s_pos, (int, float)):
                m._cusum._s_pos = float(s_pos)
            if isinstance(s_neg, (int, float)):
                m._cusum._s_neg = float(s_neg)
            if isinstance(n, int):
                m._cusum._n = n

        return m

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _call_predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return predict_proba output, or None if the model lacks it."""
        if isinstance(self._model, _PredictProbaModel):
            return np.asarray(self._model.predict_proba(X))
        if callable(self._model) and hasattr(self._model, "predict_proba"):
            fn = getattr(self._model, "predict_proba")
            return np.asarray(fn(X))
        return None

    def _call_predict(self, X: np.ndarray) -> np.ndarray:
        """Return predict output (class labels)."""
        if isinstance(self._model, _PredictModel):
            return np.asarray(self._model.predict(X))
        if callable(self._model):
            return np.asarray(self._model(X))
        raise TypeError(
            f"model must have predict_proba or predict, or be callable; "
            f"got {type(self._model)}"
        )
