from __future__ import annotations

from typing import Any, Optional

from model_monitor.core.model_actions import ModelAction
from model_monitor.storage.model_store import ModelStore
from model_monitor.training.retrain_pipeline import RetrainPipeline, RetrainResult


class ModelActionExecutor:
    """
    Executes model lifecycle actions.

    Side-effect layer:
    - model persistence
    - promotion
    - rollback
    """

    def __init__(
        self,
        *,
        model_store: ModelStore,
        retrain_pipeline: RetrainPipeline,
        dry_run: bool = False,
    ):
        self.store = model_store
        self.retrain_pipeline = retrain_pipeline
        self.dry_run = dry_run

    def execute(
        self,
        *,
        action: ModelAction,
        context: dict[str, Any],
    ) -> Optional[str]:

        if action in {ModelAction.NONE, ModelAction.REJECT}:
            return None

        if action == ModelAction.ROLLBACK:
            version = context.get("version")
            if not version:
                raise ValueError("Rollback requires target version")
            return None if self.dry_run else self.store.rollback(version)

        if action == ModelAction.PROMOTE:
            metrics = context.get("metrics", {})
            return None if self.dry_run else self.store.promote_candidate(metrics)

        if action == ModelAction.RETRAIN:
            if self.dry_run:
                return None

            retrain_df = context["retrain_df"]
            min_gain = context["min_f1_improvement"]

            try:
                current_model = self.store.load_current()
            except FileNotFoundError:
                current_model = None

            result: RetrainResult = self.retrain_pipeline.run(
                retrain_df=retrain_df,
                current_model=current_model,
                min_f1_improvement=min_gain,
            )

            if result.candidate_model is None:
                return None

            self.store.save_candidate(result.candidate_model)

            if result.promotion.promoted:
                return self.store.promote_candidate(
                    metrics={
                        "candidate_f1": result.promotion.candidate_f1,
                        "current_f1": result.promotion.current_f1,
                        "improvement": result.promotion.improvement,
                        "n_samples": result.n_samples,
                    }
                )

            return None

        raise ValueError(f"Unsupported action: {action}")
