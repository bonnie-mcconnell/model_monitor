"""Promotion decision: compare candidate vs current model F1."""
from __future__ import annotations

from dataclasses import dataclass

# Floating point subtraction is not exact: 0.82 - 0.80 evaluates to
# 0.019999999999999907 in IEEE 754. Without a tolerance, a candidate whose
# F1 improves by exactly min_improvement would be silently rejected.
_IMPROVEMENT_EPS = 1e-9


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    reason: str
    current_f1: float
    candidate_f1: float
    improvement: float


def compare_models(
    *,
    current_f1: float,
    candidate_f1: float,
    min_improvement: float,
) -> PromotionResult:
    """
    Decide whether to promote a candidate model.

    Promotion rule:
    - Candidate must outperform current model
    - Improvement must meet or exceed configured threshold

    An epsilon tolerance (_IMPROVEMENT_EPS) is applied to the threshold
    comparison to handle floating point rounding in F1 arithmetic.
    """
    improvement = candidate_f1 - current_f1

    if improvement >= min_improvement - _IMPROVEMENT_EPS:
        return PromotionResult(
            promoted=True,
            reason="candidate_outperforms_current",
            current_f1=current_f1,
            candidate_f1=candidate_f1,
            improvement=improvement,
        )

    return PromotionResult(
        promoted=False,
        reason="insufficient_improvement",
        current_f1=current_f1,
        candidate_f1=candidate_f1,
        improvement=improvement,
    )
