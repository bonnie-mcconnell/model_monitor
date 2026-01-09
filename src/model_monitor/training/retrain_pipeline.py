from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from typing import Any

from model_monitor.training.train import train_model
from model_monitor.training.evaluation import validate_model
from model_monitor.training.promotion import PromotionResult, compare_models


@dataclass(frozen=True)
class RetrainResult:
    candidate_model: Any
    promotion: PromotionResult
    n_samples: int


class RetrainPipeline:
    """
    Stateless retraining job.

    Responsibilities:
    - Train candidate model
    - Evaluate candidate vs current
    - Decide promotion eligibility
    """

    def run(
        self,
        *,
        retrain_df: pd.DataFrame,
        current_model: Any | None,
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
                candidate_model=None,
                promotion=promotion,
                n_samples=0,
            )

        candidate_model = train_model(retrain_df)
        candidate_f1 = validate_model(candidate_model, retrain_df)

        if current_model is not None:
            current_f1 = validate_model(current_model, retrain_df)
        else:
            current_f1 = 0.0

        promotion = compare_models(
            current_f1=current_f1,
            candidate_f1=candidate_f1,
            min_improvement=min_f1_improvement,
        )

        return RetrainResult(
            candidate_model=candidate_model,
            promotion=promotion,
            n_samples=len(retrain_df),
        )
