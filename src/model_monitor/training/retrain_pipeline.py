from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from model_monitor.training.train import train_model
from model_monitor.training.evaluation import validate_model
from model_monitor.training.promotion import PromotionResult, compare_models
from model_monitor.storage.model_store import ModelStore


@dataclass
class RetrainResult:
    promotion: PromotionResult
    candidate_f1: float
    current_f1: float
    had_existing_model: bool
    n_samples: int


class RetrainPipeline:
    """
    Executes retraining, evaluation, and (optional) promotion.

    Explicit dependencies:
    - ModelStore is injected to avoid hidden global state
    """

    def __init__(self, model_store: ModelStore):
        self.model_store = model_store

    def run(
        self,
        retrain_df: pd.DataFrame,
        *,
        min_f1_improvement: float,
    ) -> RetrainResult:
        if retrain_df.empty:
            promotion = PromotionResult(
                promoted=False,
                reason="empty_retrain_dataset",
                current_f1=0.0,
                candidate_f1=0.0,
                improvement=0.0,
            )
            return RetrainResult(
                promotion=promotion,
                candidate_f1=0.0,
                current_f1=0.0,
                had_existing_model=False,
                n_samples=0,
            )

        n_samples = len(retrain_df)

        candidate_model = train_model(retrain_df)
        candidate_f1 = validate_model(candidate_model, retrain_df)

        had_existing_model = True
        try:
            current_model = self.model_store.load_current()
            current_f1 = validate_model(current_model, retrain_df)
        except FileNotFoundError:
            current_f1 = 0.0
            had_existing_model = False

        promotion = compare_models(
            current_f1=current_f1,
            candidate_f1=candidate_f1,
            min_improvement=min_f1_improvement,
        )

        self.model_store.save_candidate(candidate_model)

        if promotion.promoted:
            self.model_store.promote_candidate(
                metrics={
                    "current_f1": promotion.current_f1,
                    "candidate_f1": promotion.candidate_f1,
                    "improvement": promotion.improvement,
                    "n_samples": n_samples,
                }
            )

        return RetrainResult(
            promotion=promotion,
            candidate_f1=candidate_f1,
            current_f1=current_f1,
            had_existing_model=had_existing_model,
            n_samples=n_samples,
        )
