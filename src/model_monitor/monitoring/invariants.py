from __future__ import annotations

import math
from typing import Mapping


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
        raise InvariantViolation(
            f"{name} must be in [{lo}, {hi}], got {value}"
        )


def assert_monotonic(name: str, value: int) -> None:
    if value < 0:
        raise InvariantViolation(f"{name} must be non-negative, got {value}")


def validate_trust_components(components: Mapping[str, float]) -> None:
    for key, value in components.items():
        assert_bounded(f"trust_component.{key}", value, lo=0.0, hi=1.0)
