from __future__ import annotations

from typing import Optional, Any

from model_monitor.core.decisions import Decision
from model_monitor.core.model_actions import ModelAction
from model_monitor.core.model_action_executor import ModelActionExecutor


class ModelTransitionManager:
    """
    Orchestrates transitions from decisions to model actions.
    """

    def __init__(self, executor: ModelActionExecutor):
        self.executor = executor

    def apply(
        self,
        *,
        decision: Decision,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        action = ModelAction(decision.action)
        safe_context: dict[str, Any] = context or {}
        return self.executor.execute(action=action, context=safe_context)
