from model_monitor.contracts.policy import DecisionPolicy
from model_monitor.contracts.behavioral.evaluation import GuaranteeEvaluation
from model_monitor.contracts.outcome import DecisionOutcome, OutcomeReason
from model_monitor.contracts.guarantee import Severity


class StrictBehaviorPolicy:
    policy_id = "strict_behavior_v1"

    def decide(self, *, guarantees):
        failures = [g for g in guarantees if not g.passed]

        if any(g.severity == Severity.CRITICAL for g in failures):
            return (
                DecisionOutcome.BLOCK,
                (OutcomeReason(
                    code="critical_violation",
                    message="Critical behavioral guarantee failed"
                ),),
            )

        if sum(g.severity >= Severity.HIGH for g in failures) >= 2:
            return (
                DecisionOutcome.WARN,
                (OutcomeReason(
                    code="multiple_high_failures",
                    message="Multiple high-severity behavioral regressions detected"
                ),),
            )

        return DecisionOutcome.ACCEPT, ()
