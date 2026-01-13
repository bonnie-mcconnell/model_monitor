from __future__ import annotations

from typing import Any, Optional

from model_monitor.core.model_actions import ModelAction
from model_monitor.core.decisions import Decision
from model_monitor.storage.model_store import ModelStore
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.training.retrain_pipeline import RetrainPipeline, RetrainResult


class DefaultModelActionExecutor:
    """
    Executes model lifecycle actions in a crash-safe manner.

    Guarantees:
    - No partial promotions
    - Failures are always recorded
    - Side effects are isolated per action
    """

    def __init__(
        self,
        *,
        model_store: ModelStore,
        retrain_pipeline: RetrainPipeline,
        decision_store: DecisionStore,
        dry_run: bool = False,
    ) -> None:
        self.store = model_store
        self.retrain_pipeline = retrain_pipeline
        self.decision_store = decision_store
        self.dry_run = dry_run

    def execute(
        self,
        *,
        action: ModelAction,
        context: dict[str, Any],
    ) -> Optional[str]:
        recent = self.decision_store.tail(limit=20)

        if any(
            r.action == action.value
            and r.model_version == self.store.get_active_version()
            for r in recent
        ):
            return None

        try:
            return self._execute_internal(action=action, context=context)

        except Exception as exc:
            failed_decision = Decision(
                action="system_error",
                reason=f"executor failure: {type(exc).__name__}",
                metadata={},
            )

            self.decision_store.record(
                decision=failed_decision,
                model_version=self.store.get_active_version(),
            )
            raise

    # --------------------------------------------------

    def _execute_internal(
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

            if not result.promotion.promoted:
                return None

            return self.store.promote_candidate(
                metrics={
                    "candidate_f1": result.promotion.candidate_f1,
                    "current_f1": result.promotion.current_f1,
                    "improvement": result.promotion.improvement,
                    "n_samples": result.n_samples,
                }
            )

        raise ValueError(f"Unsupported action: {action}")
