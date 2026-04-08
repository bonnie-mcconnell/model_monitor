"""Protocol defining the interface for model action executors."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

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
    ) -> str | None:
        ...
