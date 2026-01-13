from __future__ import annotations

from typing import Protocol, Mapping, Any, Optional

from model_monitor.core.model_actions import ModelAction


class ModelActionExecutorProtocol(Protocol):
    """
    Structural contract for model action execution.

    Any executor that implements this interface can be used
    by the DecisionExecutor (including test doubles).
    """

    def execute(
        self,
        *,
        action: ModelAction,
        context: Mapping[str, Any],
    ) -> Optional[str]:
        ...
