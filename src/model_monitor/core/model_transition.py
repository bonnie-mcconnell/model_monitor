from __future__ import annotations

from typing import Optional

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
        context: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Apply a decision as a model lifecycle transition.
        """
        action = ModelAction(decision.action)
        return self.executor.execute(action=action, context=context)
