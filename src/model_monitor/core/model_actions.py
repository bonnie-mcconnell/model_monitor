from __future__ import annotations

from enum import Enum


class ModelAction(str, Enum):
    """
    Executable model lifecycle actions.
    """

    NONE = "none"
    RETRAIN = "retrain"
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    REJECT = "reject"
