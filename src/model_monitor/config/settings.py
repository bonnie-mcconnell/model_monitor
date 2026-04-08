"""Application configuration loaded from YAML files."""
from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# -------------------------------
# Config schemas
# -------------------------------

class DriftConfig(BaseModel):
    """PSI drift detection thresholds loaded from config/drift.yaml."""
    psi_threshold: float
    window: int


class RetrainConfig(BaseModel):
    """Retraining policy parameters loaded from config/retrain.yaml."""
    min_f1_gain: float
    cooldown_batches: int
    min_samples: int
    min_stable_batches: int = 5


class RollbackConfig(BaseModel):
    """Rollback trigger threshold. Hardcoded default; overridable via YAML."""
    max_f1_drop: float = 0.15


class ModelConfig(BaseModel):
    """Optional model metadata from config/model.yaml."""
    name: str
    version: str
    framework: str


class AppConfig(BaseModel):
    """Fully-loaded application configuration. Immutable after construction."""
    drift: DriftConfig
    retrain: RetrainConfig
    rollback: RollbackConfig
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
    """
    Resolve config files bundled inside the model_monitor package.
    """
    resource = files("model_monitor.config").joinpath(filename)
    with as_file(resource) as path:
        return path


def load_config(
    *,
    drift_path: Path | None = None,
    retrain_path: Path | None = None,
    model_path: Path | None = None,
) -> AppConfig:
    """
    Load application configuration.
    """
    drift_path = drift_path or _default_config_path("drift.yaml")
    retrain_path = retrain_path or _default_config_path("retrain.yaml")
    model_path = model_path or _default_config_path("model.yaml")

    drift_raw = _load_yaml(drift_path)["drift"]
    retrain_raw = _load_yaml(retrain_path)["retrain"]

    model_cfg = None
    if model_path.exists():
        model_raw = _load_yaml(model_path).get("model")
        if model_raw:
            model_cfg = ModelConfig(**model_raw)

    return AppConfig(
        drift=DriftConfig(**drift_raw),
        retrain=RetrainConfig(**retrain_raw),
        rollback=RollbackConfig(),
        model=model_cfg,
    )


def load_drift_config() -> DriftConfig:
    """
    Convenience loader for inference / monitoring paths.
    """
    return load_config().drift
