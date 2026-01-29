from model_monitor.contracts.behavioral.records import DecisionRecord


def diff_decisions(
    *,
    previous: DecisionRecord,
    current: DecisionRecord,
) -> dict[str, tuple[object, object]]:
    """
    Deterministically diff two decisions.
    Used for regression detection and audits.
    """
    diffs = {}

    if previous.outcome != current.outcome:
        diffs["outcome"] = (previous.outcome, current.outcome)

    if previous.reasons != current.reasons:
        diffs["reasons"] = (previous.reasons, current.reasons)

    return diffs
