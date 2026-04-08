"""
Drift simulation with live terminal output.

Drives 80 batches through the inference pipeline. Drift is injected at
batch 40 by shifting the feature distribution. The output shows each
decision alongside the trust score and drift level so the monitoring
response is visible in real time.

Run with:
    python -m model_monitor.scripts.simulation_loop
    make sim
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from model_monitor.config.logging_config import setup_logging
from model_monitor.config.settings import AppConfig, load_config
from model_monitor.inference.predict import Predictor
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.storage.model_store import ModelStore, get_active_version
from model_monitor.training.retrain_pipeline import RetrainPipeline

setup_logging()
logger = logging.getLogger(__name__)

# Terminal colours - disabled automatically when not a tty
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _action_colour(action: str) -> str:
    if action in {"reject", "rollback"}:
        return _RED
    if action in {"retrain", "promote"}:
        return _YELLOW
    return _DIM


def _load_feature_names(base: Path | None = None) -> list[str]:
    path = (base or Path("data/reference")) / "feature_schema.json"
    with path.open() as f:
        result: list[str] = json.load(f)
        return result


def simulate_stream(
    *,
    config: AppConfig,
    n_batches: int = 80,
    batch_size: int = 64,
    drift_at_batch: int = 40,
    drift_magnitude: float = 2.5,
    label_delay: int = 2,
) -> None:
    """
    Run a drift simulation with live terminal output.

    Args:
        n_batches:       total batches to simulate
        batch_size:      samples per batch
        drift_at_batch:  batch index at which feature distribution shifts
        drift_magnitude: standard deviations of mean shift at drift point
        label_delay:     batches before delayed labels become available
    """
    if config.model is None:
        raise ValueError("Simulation requires model configuration (config/model.yaml)")

    FEATURE_NAMES = _load_feature_names()

    model_store   = ModelStore()
    predictor     = Predictor(config=config)
    retrain_buffer = RetrainEvidenceBuffer(min_samples=config.retrain.min_samples)
    retrain_pipeline = RetrainPipeline(model_store=model_store)

    print(f"\n{_BOLD}━━━  model_monitor simulation  ━━━{_RESET}")
    print(f"  {n_batches} batches · drift injected at batch {drift_at_batch}")
    print(f"  {'batch':>6}  {'drift':>7}  {'trust':>7}  {'action'}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*20}")

    for step in range(n_batches):
        batch_id = f"batch_{step}_{uuid.uuid4().hex[:6]}"
        drifted  = step >= drift_at_batch
        shift    = drift_magnitude if drifted else 0.0

        X = np.random.default_rng(step).normal(
            loc=shift, scale=1.0, size=(batch_size, len(FEATURE_NAMES))
        )
        X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
        y    = (X_df[FEATURE_NAMES[0]] > 0).astype(int)

        preds, confs, decision = predictor.predict_batch(X_df, y_true=y, batch_id=batch_id)

        drift_score = cast(float, decision.metadata.get("drift_score", 0.0))
        trust_score = cast(float, decision.metadata.get("trust_score", 1.0))

        retrain_buffer.add_summary(
            accuracy=float((preds == y).mean()),
            f1=cast(float, decision.metadata.get("f1", 0.0)),
            drift_score=drift_score,
            trust_score=trust_score,
            timestamp=time.time(),
        )

        colour = _action_colour(decision.action)
        marker = " ← drift" if step == drift_at_batch else ""
        print(
            f"  {step:>6}  {drift_score:>7.3f}  {trust_score:>7.3f}  "
            f"{colour}{decision.action:<12}{_RESET}{_DIM}{marker}{_RESET}"
        )

        # Structural log for downstream consumers
        logger.info(
            "batch_decision",
            extra={
                "batch_id": batch_id,
                "step": step,
                "drifted": drifted,
                "drift_score": round(drift_score, 4),
                "trust_score": round(trust_score, 4),
                "action": decision.action,
                "model_version": get_active_version(),
            },
        )

        if decision.action == "rollback":
            target = decision.metadata.get("target_version")
            if target is None:
                logger.error("rollback_missing_target_version", extra={"batch_id": batch_id})
            else:
                model_store.rollback(version=str(target))
                predictor.reload()
                print(f"  {_YELLOW}↩  rolled back to {target}{_RESET}")

        if decision.action == "retrain" and retrain_buffer.ready():
            retrain_df = retrain_buffer.consume()
            result = retrain_pipeline.run(retrain_df, min_f1_improvement=config.retrain.min_f1_gain)
            if result.promotion.promoted:
                predictor.reload()
                print(f"  {_GREEN}↑  new model promoted (F1 {result.promotion.candidate_f1:.3f}){_RESET}")
            else:
                print(f"  {_DIM}↷  candidate rejected (insufficient improvement){_RESET}")

        time.sleep(0.02)

    print(f"\n{_DIM}  simulation finished - {n_batches} batches{_RESET}\n")


def main() -> None:  # pragma: no cover
    cfg = load_config(
        drift_path=Path("config/drift.yaml"),
        retrain_path=Path("config/retrain.yaml"),
        model_path=Path("config/model.yaml"),
    )
    simulate_stream(config=cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
