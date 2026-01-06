from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from model_monitor.training.train import train_model
from model_monitor.training.evaluation import validate_model
from model_monitor.training.promotion import PromotionResult, compare_models
from model_monitor.storage import model_store


@dataclass
class RetrainResult:
    """
    Structured outcome of a retraining attempt.

    This is intentionally verbose to make retraining decisions
    auditable, testable, and easy to inspect downstream
    (dashboards, logs, decision history, etc.).
    """
    promotion: PromotionResult
    candidate_f1: float
    current_f1: float
    had_existing_model: bool
    n_samples: int


class RetrainPipeline:
    """
    Executes retraining, evaluation, and (optional) promotion.

    Responsibilities:
    - Train a candidate model from aggregated retraining evidence
    - Evaluate candidate vs current model on the same dataset
    - Persist the candidate model for auditability
    - Promote the candidate if it meets promotion criteria

    Non-responsibilities:
    - Deciding *when* to retrain
    - Collecting raw data
    - Managing thresholds or alerting

    This separation is deliberate and mirrors production ML systems.
    """

    def run(
        self,
        retrain_df: pd.DataFrame,
        *,
        min_f1_improvement: float,
    ) -> RetrainResult:
        # --- Guard: empty retrain request ---
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

        # --- Train candidate model ---
        candidate_model = train_model(retrain_df)
        candidate_f1 = validate_model(candidate_model, retrain_df)

        # --- Load & evaluate current model (if any) ---
        had_existing_model = True
        try:
            current_model = model_store.load_current()
            current_f1 = validate_model(current_model, retrain_df)
        except FileNotFoundError:
            current_f1 = 0.0
            had_existing_model = False

        # --- Decide promotion ---
        promotion = compare_models(
            current_f1=current_f1,
            candidate_f1=candidate_f1,
            min_improvement=min_f1_improvement,
        )

        # --- Persist candidate model (always) ---
        # We always save the candidate so decisions are auditable,
        # even if promotion is rejected.
        model_store.save_candidate(candidate_model)

        # --- Promote if approved ---
        if promotion.promoted:
            model_store.promote_candidate(
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
