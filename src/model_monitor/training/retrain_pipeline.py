"""End-to-end retraining: train, evaluate on held-out split, decide promotion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from model_monitor.storage.model_store import ModelStore
from model_monitor.training.evaluation import validate_model
from model_monitor.training.promotion import PromotionResult, compare_models
from model_monitor.training.train import train_model

# Fraction of retrain data held out for validation.
# Kept constant so promotion decisions are evaluated on the same
# distribution in every run - not on the training samples the
# candidate already saw.
_VAL_FRACTION = 0.2
_VAL_MIN_SAMPLES = 50  # below this, skip the split and use full dataset


@dataclass(frozen=True)
class RetrainResult:
    candidate_model: Any
    promotion: PromotionResult
    n_samples: int


class RetrainPipeline:
    """
    Stateless retraining job.

    Responsibilities:
    - Split retrain data into train / validation sets
    - Train candidate model on training split
    - Evaluate both candidate and current model on the held-out validation set
    - Decide promotion eligibility based on the comparison

    Validation split:
        A held-out validation set (_VAL_FRACTION = 20%) is used for evaluation.
        Without this, candidate F1 is measured on the same data the model was
        trained on - producing optimistic in-sample estimates that favour
        promotion of overfit models. The current model is evaluated on the same
        validation split so the comparison is fair.

        When retrain_df is too small (< _VAL_MIN_SAMPLES rows total), the split
        is skipped and the full dataset is used for both training and evaluation.
        This is a deliberate trade-off: very small datasets make a 20% split
        statistically unreliable, and the min_samples guard on RetrainEvidenceBuffer
        is the primary protection against noisy retraining signals.
    """

    def __init__(self, *, model_store: ModelStore) -> None:
        self.model_store = model_store

    def run(
        self,
        retrain_df: pd.DataFrame,
        current_model: Any | None = None,
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
                candidate_model=None,
                promotion=promotion,
                n_samples=0,
            )

        n = len(retrain_df)

        if n >= _VAL_MIN_SAMPLES:
            train_df, val_df = train_test_split(
                retrain_df,
                test_size=_VAL_FRACTION,
                random_state=42,
                shuffle=True,
            )
        else:
            # Too few samples for a reliable split - use full dataset.
            # In-sample evaluation is biased but preferable to a val set
            # of only a handful of rows.
            train_df = retrain_df
            val_df = retrain_df

        candidate_model = train_model(train_df)

        # Both models evaluated on the same held-out set for a fair comparison.
        candidate_f1 = validate_model(candidate_model, val_df)

        if current_model is not None:
            current_f1 = validate_model(current_model, val_df)
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
            n_samples=n,
        )
