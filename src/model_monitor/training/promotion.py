from dataclasses import dataclass


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
    - Improvement must exceed configured threshold
    """
    improvement = candidate_f1 - current_f1

    if improvement >= min_improvement:
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
