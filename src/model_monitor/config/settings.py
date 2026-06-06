"""Application configuration loaded from YAML files."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator

# -------------------------------
# Config schemas
# -------------------------------


class DriftConfig(BaseModel):
    """PSI drift detection thresholds loaded from config/drift.yaml."""

    psi_threshold: float
    window: int


class RetrainConfig(BaseModel):
    """Retraining policy parameters loaded from config/retrain.yaml.

    Attributes:
        min_f1_gain:        Minimum absolute F1 drop from baseline that triggers
                            a retrain decision in the engine.
        cooldown_batches:   Minimum batches between two retrain attempts
                            (prevents rapid cycling on noisy data).
        min_samples:        Minimum raw labeled rows the RawDataBuffer must hold
                            before a new model is actually trained.  Below this
                            threshold the executor falls back to synthetic data.
        evidence_window:    Minimum number of aggregation-loop summaries that
                            must accumulate before the DecisionExecutor will
                            act on a retrain decision.  This is intentionally
                            small (default 3) so the monitor reacts within
                            three aggregation cycles rather than the hundreds
                            that ``min_samples`` would imply if used here.
                            In the simulation each batch produces one summary,
                            so ``evidence_window=3`` means three consecutive
                            degraded batches suffice to trigger a retrain.
        min_stable_batches: Consecutive ``none``-action batches required before
                            the engine fires a ``promote`` decision.
        max_retrain_attempts: Maximum number of retrain decisions the engine
                            may fire before it escalates to ``system_error`` and
                            halts further retraining.  This is the circuit breaker:
                            if every retrain produces a model that still drifts,
                            the system would otherwise retry indefinitely, burning
                            compute and masking the underlying data problem.
                            When the circuit breaker fires the monitor emits
                            ``system_error`` until an operator resets the counter
                            by calling ``engine.reset_retrain_counter()``.
                            Default 10 - generous enough for normal operation
                            (drift events → retrain → stabilise is one cycle)
                            while preventing runaway loops in broken pipelines.
                            Set to 0 to disable the circuit breaker entirely.
    """

    min_f1_gain: float
    cooldown_batches: int
    min_samples: int
    evidence_window: int = 3
    min_stable_batches: int = 5
    max_retrain_attempts: int = 10


class RollbackConfig(BaseModel):
    """Rollback trigger threshold. Hardcoded default; overridable via YAML."""

    max_f1_drop: float = 0.15


class ModelConfig(BaseModel):
    """Optional model metadata from config/model.yaml."""

    name: str
    version: str
    framework: str


class TrustScoreConfig(BaseModel):
    """Trust score component weights loaded from config/trust_score.yaml.

    All seven weights must sum to 1.0 (within floating-point tolerance).
    The constraint is validated at startup - misconfigured deployments fail
    loudly rather than silently producing wrong scores.

    Component definitions:
        accuracy     - batch accuracy_score vs ground truth labels (25%*)
        f1           - batch F1 score, macro averaged (20%*)
        calibration  - Expected Calibration Error → [0,1]. Replaces 'confidence'.
                       ECE=0 → 1.0 | ECE=0.05 → 0.5 | ECE≥0.10 → 0.0
        drift        - max(input PSI, output PSI) → [0,1]
        latency      - p95 per-sample latency ms → [0,1]
        data_quality - null rate + range violations + schema consistency → [0,1]
        behavioral   - contract violation EMA (BM branch; no-op on main)

    v5 migration: 'confidence' was renamed to 'calibration'. Update YAML files.
    """

    accuracy: float = 0.23
    f1: float = 0.18
    calibration: float = 0.14
    drift: float = 0.18
    latency: float = 0.17
    data_quality: float = 0.05
    behavioral: float = 0.05

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> TrustScoreConfig:
        total = (
            self.accuracy
            + self.f1
            + self.calibration
            + self.drift
            + self.latency
            + self.data_quality
            + self.behavioral
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Trust score weights must sum to 1.0, got {total:.6f}. "
                f"Got: accuracy={self.accuracy}, f1={self.f1}, "
                f"calibration={self.calibration}, drift={self.drift}, "
                f"latency={self.latency}, data_quality={self.data_quality}, "
                f"behavioral={self.behavioral}. Check config/trust_score.yaml."
            )
        return self


class AppConfig(BaseModel):
    """Fully-loaded application configuration. Immutable after construction."""

    drift: DriftConfig
    retrain: RetrainConfig
    rollback: RollbackConfig
    trust_score: TrustScoreConfig = TrustScoreConfig()
    model: ModelConfig | None = None


# -------------------------------
# Loader helpers
# -------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file. Raises FileNotFoundError if missing."""
    from model_monitor.utils.io import load_yaml as _load

    result = _load(path)
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML dict at {path}, got {type(result).__name__}")
    return result


def _default_config_path(filename: str) -> Path:
    """Resolve config files bundled inside the model_monitor package."""
    resource = files("model_monitor.config").joinpath(filename)
    with as_file(resource) as path:
        return path


def load_config(
    *,
    drift_path: Path | None = None,
    retrain_path: Path | None = None,
    rollback_path: Path | None = None,
    trust_score_path: Path | None = None,
    model_path: Path | None = None,
) -> AppConfig:
    """Load application configuration from YAML files."""
    drift_path = drift_path or _default_config_path("drift.yaml")
    retrain_path = retrain_path or _default_config_path("retrain.yaml")
    rollback_path = rollback_path or _default_config_path("rollback.yaml")
    trust_score_path = trust_score_path or _default_config_path("trust_score.yaml")
    model_path = model_path or _default_config_path("model.yaml")

    drift_raw = _load_yaml(drift_path)["drift"]
    retrain_raw = _load_yaml(retrain_path)["retrain"]

    # RollbackConfig falls back to its hardcoded default when the file is
    # absent so that older deployments lacking rollback.yaml keep working.
    rollback_cfg = RollbackConfig()
    if rollback_path.exists():
        rollback_raw = _load_yaml(rollback_path).get("rollback")
        if rollback_raw:
            rollback_cfg = RollbackConfig(**rollback_raw)

    # TrustScoreConfig falls back to defaults when the file is absent.
    # This preserves backward compatibility with deployments that have not
    # yet added trust_score.yaml.
    trust_score_cfg = TrustScoreConfig()
    if trust_score_path.exists():
        ts_raw = _load_yaml(trust_score_path).get("trust_score")
        if ts_raw:
            trust_score_cfg = TrustScoreConfig(**ts_raw)

    model_cfg = None
    if model_path.exists():
        model_raw = _load_yaml(model_path).get("model")
        if model_raw:
            model_cfg = ModelConfig(**model_raw)

    return AppConfig(
        drift=DriftConfig(**drift_raw),
        retrain=RetrainConfig(**retrain_raw),
        rollback=rollback_cfg,
        trust_score=trust_score_cfg,
        model=model_cfg,
    )


def load_drift_config() -> DriftConfig:
    """Convenience loader for inference / monitoring paths."""
    return load_config().drift
