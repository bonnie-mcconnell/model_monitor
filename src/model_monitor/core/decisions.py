"""Core domain types: DecisionType, DecisionMetadata, Decision.

DecisionType is a StrEnum whose values ARE plain strings, so every existing
string comparison, SQLite storage, JSON serialisation and Prometheus label
continues to work without modification at call sites:

    decision.action == "retrain"  # True - StrEnum value is the string itself
    json.dumps(decision.action)   # '"retrain"'
    f"action={decision.action}"   # 'action=retrain'

The Enum adds:
  - Tab-completion and IDE navigation to every usage
  - Membership testing:  ``DecisionType.RETRAIN in allowed``
  - Iteration:           ``list(DecisionType)`` enumerates all valid actions
  - Typo safety:         ``DecisionType("retrainX")`` raises ValueError
  - Single source of truth: adding a new action only requires a change here
    (previously split across decisions.py, dashboard.py, schemas.py, and
    string literals scattered through storage, simulation, and CLI code)

Python compatibility:
  ``StrEnum`` was added in Python 3.11.  We provide a minimal compatibility
  shim so the code runs on Python 3.10 as well.  The shim behaves identically
  to the stdlib ``StrEnum`` for all operations used in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict, cast

try:
    from enum import StrEnum as _StrEnumBase  # Python 3.11+
except ImportError:  # pragma: no cover - only reached on Python 3.10

    class _StrEnumBase(str, Enum):  # type: ignore[no-redef]
        """Minimal StrEnum shim for Python 3.10."""

        def __str__(self) -> str:
            return str(self.value)


class DecisionType(_StrEnumBase):
    """Exhaustive set of operational decisions the engine can produce.

    Each member's value is the canonical lowercase string stored in SQLite,
    returned in API responses, used as Prometheus label values, and written
    in CLI output.

    Adding a new action type:
      1. Add the member here.
      2. Handle it in ``DecisionEngine.decide()`` - add the rule.
      3. Handle it in ``DecisionExecutor.execute()`` - add the side-effect.
      4. Add a colour to ``ACTION_COLOURS`` in ``streamlit_app.py``.
    """

    NONE = "none"
    RETRAIN = "retrain"
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    REJECT = "reject"
    SYSTEM_ERROR = "system_error"


class DecisionMetadata(TypedDict, total=False):
    trust_score: float
    f1_drop: float
    baseline_f1: float
    current_f1: float
    drift_score: float
    threshold: float
    stable_batches: int
    cooldown_batches: int
    batches_since_last_retrain: int
    consecutive_rejects: int

    # Diagnostic / gating metadata
    has_labels: bool
    has_baseline: bool
    n_samples: int

    # Circuit breaker state - populated on RETRAIN and SYSTEM_ERROR decisions.
    retrain_attempt_count: int
    max_retrain_attempts: int

    # Audit trail: links the decision to the batch that produced it
    batch_id: str


@dataclass(frozen=True)
class Decision:
    """
    Represents a system-level operational decision.

    Lightweight, immutable, JSON-serializable.
    ``action`` is a ``DecisionType`` member; since ``DecisionType`` is a
    ``StrEnum``, it serialises as a plain string in JSON and compares equal
    to the equivalent string literal (``decision.action == "retrain"`` is
    ``True``).
    """

    action: DecisionType
    reason: str
    metadata: DecisionMetadata = field(
        default_factory=lambda: cast(DecisionMetadata, {})
    )
