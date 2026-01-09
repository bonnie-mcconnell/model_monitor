from __future__ import annotations

from typing import Any, Dict, Optional

from model_monitor.core.model_actions import ModelAction
from model_monitor.storage.model_store import ModelStore


class ModelActionExecutor:
    """
    Executes model lifecycle actions.

    This layer:
    - mutates model state
    - calls training / evaluation pipelines
    - is intentionally thin and explicit
    """

    def __init__(
        self,
        *,
        model_store: ModelStore,
        retrain_pipeline: callable,
        evaluate_model: callable,
        dry_run: bool = False,
    ):
        self.store = model_store
        self.retrain_pipeline = retrain_pipeline
        self.evaluate_model = evaluate_model
        self.dry_run = dry_run

    def execute(
        self,
        *,
        action: ModelAction,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Execute a model action.

        Returns:
            Optional[str]: model version if applicable
        """
        context = context or {}

        if action == ModelAction.NONE:
            return None

        if action == ModelAction.REJECT:
            # Explicit no-op; logged elsewhere
            return None

        if action == ModelAction.RETRAIN:
            if self.dry_run:
                return None

            model = self.retrain_pipeline(context)
            metrics = self.evaluate_model(model)

            self.store.save_candidate(model)
            return self.store.promote_candidate(metrics)

        if action == ModelAction.PROMOTE:
            if self.dry_run:
                return None

            return self.store.promote_candidate(context.get("metrics"))

        if action == ModelAction.ROLLBACK:
            if self.dry_run:
                return None

            version = context.get("version")
            if not version:
                raise ValueError("Rollback requires a target version")

            return self.store.rollback(version)

        raise ValueError(f"Unsupported action: {action}")
