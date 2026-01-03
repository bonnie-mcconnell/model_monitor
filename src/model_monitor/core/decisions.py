from dataclasses import dataclass
from typing import Literal, Dict, Any

DecisionType = Literal["none", "retrain", "promote", "rollback", "reject"]


@dataclass(frozen=True)
class Decision:
    """
    Represents a system-level operational decision.

    This object is intentionally lightweight and
    JSON-serializable so it can be:
    - logged
    - persisted
    - returned via APIs
    """

    action: DecisionType
    reason: str
    metadata: Dict[str, Any]
