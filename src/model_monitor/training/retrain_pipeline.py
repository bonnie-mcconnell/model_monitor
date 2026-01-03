from dataclasses import dataclass

import pandas as pd

from model_monitor.training.train import train_model
from model_monitor.training.evaluation import validate_model
from model_monitor.training.promotion import PromotionResult, compare_models
from model_monitor.storage import model_store


@dataclass
class RetrainResult:
    promotion: PromotionResult


class RetrainPipeline:
    """
    Executes retraining, evaluation, and promotion.

    This pipeline does NOT decide *when* to retrain.
    It only executes a retrain request once triggered.
    """

    def run(
        self,
        retrain_df: pd.DataFrame,
        *,
        min_f1_improvement: float,
    ) -> RetrainResult:
        if retrain_df.empty:
            return RetrainResult(
                promotion=PromotionResult(
                    promoted=False,
                    reason="empty_retrain_dataset",
                    current_f1=0.0,
                    candidate_f1=0.0,
                    improvement=0.0,
                )
            )

        # --- Train candidate model ---
        candidate_model = train_model(retrain_df)
        candidate_f1 = validate_model(candidate_model, retrain_df)

        # --- Evaluate current model (if any) ---
        try:
            current_model = model_store.load_current()
            current_f1 = validate_model(current_model, retrain_df)
        except FileNotFoundError:
            current_f1 = 0.0

        # --- Decide promotion ---
        promotion = compare_models(
            current_f1=current_f1,
            candidate_f1=candidate_f1,
            min_improvement=min_f1_improvement,
        )

        # --- Persist candidate ---
        model_store.save_candidate(candidate_model)

        # --- Promote if approved ---
        if promotion.promoted:
            model_store.promote_candidate(
                metrics={
                    "current_f1": promotion.current_f1,
                    "candidate_f1": promotion.candidate_f1,
                    "improvement": promotion.improvement,
                }
            )

        return RetrainResult(promotion=promotion)
