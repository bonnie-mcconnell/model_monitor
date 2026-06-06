"""MetricRecord TypedDict - canonical monitoring record schema."""

# src/model_monitor/monitoring/types.py
from __future__ import annotations

from typing import Literal, TypedDict

from model_monitor.core.decisions import DecisionType

__all__ = ["DecisionType", "MetricName", "MetricRecord"]

MetricName = Literal[
    "accuracy",
    "f1",
    "avg_confidence",
    "drift_score",
    "decision_latency_ms",
]


class MetricRecord(TypedDict):
    """
    Canonical monitoring record.

    Represents the outcome of a decision batch and any resulting
    model action. This is the single source of truth for:
    - monitoring
    - dashboards
    - audits
    - analytics
    """

    timestamp: float
    batch_id: str
    n_samples: int

    accuracy: float
    f1: float
    avg_confidence: float
    drift_score: float
    decision_latency_ms: float

    # p95 / p99 tail latency within the batch.  Computed when the batch
    # is large enough (>= 20 samples) to give meaningful percentile estimates.
    # None on small batches or when per-sample timing is not available.
    p95_latency_ms: float | None
    p99_latency_ms: float | None

    # Expected Calibration Error: measures whether confidence scores match
    # actual accuracy.  None when y_true is not available (no labels).
    calibration_error: float | None

    # Per-feature PSI scores, one entry per feature in training order.
    # None when drift monitoring is not configured or the buffer window
    # has not yet been filled.  Serialised as a JSON array in the ORM.
    feature_drift_scores: list[float] | None

    # PSI applied to the model's output probability distribution.
    # None when OutputDriftMonitor is not configured or buffer not full.
    # Detects prediction distribution shift before performance degrades.
    output_drift_score: float | None

    # Per-class output PSI scores.  None when OutputDriftMonitor not configured.
    # Serialised as JSON array in ORM.
    output_drift_class_scores: list[float] | None

    # Scalar data quality score in [0, 1].  1.0 = no issues detected.
    # Aggregates null rate, out-of-range rate, and schema consistency checks.
    # None when DataQualityMonitor is not configured.
    data_quality_score: float | None

    # Conformal coverage rate on this batch (fraction of labeled samples
    # whose true label falls within the prediction set).
    # None when ConformalMonitor is not calibrated or labels are absent.
    conformal_coverage: float | None

    # Mean conformal prediction set size.  Growing set size signals declining
    # model confidence even when labels are unavailable.
    # None when ConformalMonitor is not configured.
    conformal_set_size: float | None

    # Exponential moving average of per-batch behavioral violation scores
    # at the time this batch was processed.  None when no BehavioralContractRunner
    # is configured.  Feeds compute_trust_score and the Prometheus endpoint.
    behavioral_violation_rate: float | None

    # Per-feature SHAP importance shift vs. training-time baseline.
    # Positive value: model relies on this feature more than at training time.
    # None when ShapDriftAttributor is not configured.
    # Serialised as a JSON object {"f0": 0.03, "f1": -0.01, ...} in the ORM.
    # Causal drift attribution report - serialised as JSON when present.
    # None when CausalDriftAttributor is not configured or no drift detected.
    causal_drift_report: dict[str, object] | None

    # MMD joint distribution drift test results.
    # mmd_p_value:  permutation test p-value (small = joint drift detected).
    # mmd_is_drift: True when p_value < alpha (default 0.05).
    # None when MMDDriftDetector is not configured, the batch had < 5 samples,
    # or this was a skipped batch (mmd_every > 1).
    mmd_p_value: float | None
    mmd_is_drift: bool | None

    shap_attribution: dict[str, float] | None

    action: DecisionType
    reason: str

    previous_model: str | None
    new_model: str | None
