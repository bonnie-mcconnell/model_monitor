"""Runtime invariant checks - raise loudly when monitoring data is corrupted."""

from __future__ import annotations

import math
from collections.abc import Mapping


class InvariantViolation(RuntimeError):
    """Raised when a system invariant is violated."""


def assert_finite(name: str, value: float) -> None:
    if math.isnan(value) or math.isinf(value):
        raise InvariantViolation(f"{name} must be finite, got {value}")


def assert_bounded(
    name: str,
    value: float,
    *,
    lo: float,
    hi: float,
) -> None:
    assert_finite(name, value)
    if not (lo <= value <= hi):
        raise InvariantViolation(f"{name} must be in [{lo}, {hi}], got {value}")


def assert_non_negative(name: str, value: int | float) -> None:
    """Raise InvariantViolation if ``value`` is negative."""
    if value < 0:
        raise InvariantViolation(f"{name} must be non-negative, got {value}")


# Stateful monotonicity checker.  A single process-level instance is used by
# the aggregation loop; tests that need isolation instantiate their own.
class MonotonicityChecker:
    """Tracks per-key last-seen values and raises if a value ever decreases.

    Monotonicity here means non-decreasing: equal values are allowed (e.g.
    a window with zero new batches produces the same n_batches twice).
    Strictly increasing sequences are a special case.

    Each key is tracked independently.  Adding a new key initialises it to
    the first observed value without raising.

    Thread safety: not thread-safe.  Intended for single-threaded asyncio use.
    """

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def check(self, name: str, value: int) -> None:
        """Raise InvariantViolation if ``value`` < previously seen value."""
        assert_non_negative(name, value)
        if name in self._seen and value < self._seen[name]:
            raise InvariantViolation(
                f"{name} decreased from {self._seen[name]} to {value}"
            )
        self._seen[name] = value

    def reset(self, name: str | None = None) -> None:
        """Clear state for one key, or all keys when ``name`` is None."""
        if name is None:
            self._seen.clear()
        else:
            self._seen.pop(name, None)


def validate_trust_components(components: Mapping[str, float]) -> None:
    for key, value in components.items():
        assert_bounded(f"trust_component.{key}", value, lo=0.0, hi=1.0)
