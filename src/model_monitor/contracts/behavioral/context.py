from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """
    Immutable envelope describing a single model interaction.
    This is the unit of truth for behavioral decisions.
    """
    run_id: str
    model_id: str
    prompt_id: str
    output: str
    metadata: dict[str, Any]