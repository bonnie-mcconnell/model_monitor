"""ORM model for batch-level metric records."""

from __future__ import annotations

from sqlalchemy import Boolean, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from model_monitor.storage.db import Base


class MetricsRecordORM(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[float] = mapped_column(Float, index=True)

    batch_id: Mapped[str] = mapped_column(String, index=True)
    n_samples: Mapped[int] = mapped_column(Integer)

    accuracy: Mapped[float] = mapped_column(Float)
    f1: Mapped[float] = mapped_column(Float)
    avg_confidence: Mapped[float] = mapped_column(Float)
    drift_score: Mapped[float] = mapped_column(Float)
    decision_latency_ms: Mapped[float] = mapped_column(Float)

    # Tail-latency percentiles - null on small batches or when per-sample
    # timing is not available.  p95 is used by the trust score instead of mean.
    p95_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    p99_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Expected Calibration Error - null when ground-truth labels were unavailable.
    calibration_error: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Per-feature PSI scores stored as JSON array.
    # Null when drift monitoring is not configured or buffer is not yet full.
    # e.g. '[0.02, 0.15, 0.01, ...]' - one float per feature, same order as
    # the training feature schema.
    feature_drift_scores: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Output (prediction) distribution PSI - scalar mean across classes.
    # Null when OutputDriftMonitor is not configured.
    output_drift_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Per-class output PSI scores stored as JSON array.
    # e.g. '[0.02, 0.15]' - one float per output class.
    output_drift_class_scores: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Aggregate data quality score in [0, 1].
    # Null when DataQualityMonitor is not configured.
    data_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Conformal coverage rate for this batch - fraction of labeled samples
    # whose true label fell within the conformal prediction set.
    # Null when ConformalMonitor is not calibrated or labels are unavailable.
    conformal_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Mean conformal prediction set size.  Growing set size signals declining
    # model confidence even without labels.
    conformal_set_size: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Exponential moving average of per-batch behavioral violation scores.
    # Null when no BehavioralContractRunner is configured.
    behavioral_violation_rate: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    # Causal drift attribution report - JSON-serialised dict.
    # Contains dominant_cause, recommendation, and per-feature classifications.
    # Null when CausalDriftAttributor not configured or no drift detected.
    causal_drift_report: Mapped[str | None] = mapped_column(Text, nullable=True)

    # MMD joint distribution drift test results.
    # mmd_p_value small (< alpha) → statistically significant joint drift.
    # mmd_is_drift True → PSI may be clean but the joint distribution shifted.
    mmd_p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    mmd_is_drift: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Per-feature SHAP importance shift stored as JSON object.
    # Null when ShapDriftAttributor is not configured.
    # e.g. '{"f0": 0.03, "f1": -0.01, ...}' - shift vs training baseline.
    shap_attribution: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Expected values align with DecisionType
    action: Mapped[str] = mapped_column(String, index=True)
    reason: Mapped[str] = mapped_column(String)

    previous_model: Mapped[str | None] = mapped_column(String, nullable=True)
    new_model: Mapped[str | None] = mapped_column(String, nullable=True)


Index("idx_metrics_action_ts", MetricsRecordORM.action, MetricsRecordORM.timestamp)
