"""
Main simulation loop.

Drives streaming inference, delayed labels, monitoring,
decision evaluation, and retraining execution.
"""
import json
import time
import uuid
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from model_monitor.config.settings import AppConfig, load_config
from model_monitor.inference.predict import Predictor
from model_monitor.monitoring.retrain_buffer import RetrainEvidenceBuffer
from model_monitor.config.logging_config import setup_logging
from model_monitor.storage.model_store import ModelStore, get_active_version
from model_monitor.training.retrain_pipeline import RetrainPipeline

setup_logging()
logger = logging.getLogger(__name__)

SCHEMA_PATH = Path("data/reference/feature_schema.json")
with SCHEMA_PATH.open() as f:
    FEATURE_NAMES: list[str] = json.load(f)


def simulate_stream(
    *,
    config: AppConfig,
    n_batches: int = 50,
    batch_size: int = 32,
    label_delay: int = 2,
    stress: bool = False,
) -> None:
    assert config.model is not None, "Simulation requires model configuration"

    # ---- Explicit lifecycle dependencies
    model_store = ModelStore()

    predictor = Predictor(config=config)

    retrain_buffer = RetrainEvidenceBuffer(
        min_samples=config.retrain.min_samples
    )

    retrain_pipeline = RetrainPipeline(
        model_store=model_store
    )

    pending_labels: list[tuple[int, pd.DataFrame]] = []

    logger.info(
        "simulation_started",
        extra={
            "batches": n_batches,
            "batch_size": batch_size,
            "stress": stress,
        },
    )

    for step in range(n_batches):
        batch_id = f"batch_{step}_{uuid.uuid4().hex[:6]}"

        shift = 2.5 if stress and step > n_batches // 2 else 0.0

        X = np.random.normal(
            loc=shift,
            scale=1.0,
            size=(batch_size, len(FEATURE_NAMES)),
        )
        X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

        y = (X_df[FEATURE_NAMES[0]] > 0).astype(int)

        labeled_df = X_df.copy()
        labeled_df["label"] = y

        start_time = time.perf_counter()

        preds, confs, decision = predictor.predict_batch(
            X_df,
            y_true=y,
            batch_id=batch_id,
        )

        decision_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # ---- Retrain evidence
        retrain_buffer.add_summary(
            accuracy=float((preds == y).mean()),
            f1=decision.metadata.get("f1", 0.0),
            drift_score=decision.metadata.get("drift_score", 0.0),
            trust_score=decision.metadata.get("trust_score", 1.0),
            timestamp=time.time(),
        )

        logger.info(
            "decision_completed",
            extra={
                "batch_id": batch_id,
                "model_version": get_active_version(),
                "decision_action": decision.action,
                "decision_latency_ms": round(decision_latency_ms, 2),
            },
        )

        # ---- Rollback handling
        if decision.action == "rollback":
            target_version = decision.metadata.get("target_version")
            if target_version is None:
                logger.error(
                    "rollback_missing_target_version",
                    extra={"batch_id": batch_id},
                )
            else:
                rolled_to = model_store.rollback(version=target_version)
                predictor.reload()
                logger.warning(
                    "model_rollback",
                    extra={
                        "batch_id": batch_id,
                        "rolled_to": rolled_to,
                        "reason": decision.reason,
                    },
                )

        # ---- Label delay (future extension)
        pending_labels.append((step + label_delay, labeled_df))
        pending_labels = [
            item for item in pending_labels if step < item[0]
        ]

        # ---- Retraining trigger
        if decision.action == "retrain" and retrain_buffer.ready():
            retrain_df = retrain_buffer.consume()
            result = retrain_pipeline.run(
                retrain_df,
                min_f1_improvement=config.retrain.min_f1_gain,
            )

            if result.promotion.promoted:
                predictor.reload()
                logger.info("model_promoted")
            else:
                logger.info("candidate_rejected")

        time.sleep(0.05)

    logger.info("simulation_finished")


if __name__ == "__main__":
    cfg = load_config(
        drift_path=Path("config/drift.yaml"),
        retrain_path=Path("config/retrain.yaml"),
        model_path=Path("config/model.yaml"),
    )

    simulate_stream(config=cfg, stress=True)
