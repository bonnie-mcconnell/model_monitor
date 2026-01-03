from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel
from importlib.resources import files, as_file


# -------------------------------
# Config schemas
# -------------------------------

class DriftConfig(BaseModel):
    psi_threshold: float
    window: int


class RetrainConfig(BaseModel):
    min_f1_gain: float
    cooldown_batches: int
    min_samples: int
    min_stable_batches: int = 5


class RollbackConfig(BaseModel):
    max_f1_drop: float = 0.15


class ModelConfig(BaseModel):
    name: str
    version: str
    framework: str


class AppConfig(BaseModel):
    drift: DriftConfig
    retrain: RetrainConfig
    rollback: RollbackConfig
    model: Optional[ModelConfig] = None


# -------------------------------
# Loader helpers
# -------------------------------

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def _default_config_path(filename: str) -> Path:
    """
    Resolve config files bundled inside the model_monitor package.
    """
    resource = files("model_monitor.config").joinpath(filename)
    with as_file(resource) as path:
        return path


def load_config(
    *,
    drift_path: Optional[Path] = None,
    retrain_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
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
