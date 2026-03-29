from __future__ import annotations

import uuid
from datetime import datetime, timezone

from model_monitor.contracts.behavioral.context import DecisionContext
from model_monitor.contracts.behavioral.evaluation import GuaranteeEvaluation
from model_monitor.contracts.behavioral.policy import DecisionPolicy
from model_monitor.contracts.behavioral.records import DecisionRecord
from model_monitor.contracts.contract import Contract
from model_monitor.contracts.outcome import DecisionOutcome
from model_monitor.contracts.registry import EvaluatorRegistry


class BehavioralContractRunner:
    """
    Evaluates a single model interaction against a behavioral contract.

    Steps:
      1. For each guarantee in the contract, look up its evaluator.
      2. Run the evaluator against the raw model output.
      3. Collect results into GuaranteeEvaluation records.
      4. Pass results to the decision policy.
      5. Return an immutable, auditable DecisionRecord.
    """

    def __init__(
        self,
        *,
        contract: Contract,
        registry: EvaluatorRegistry,
        policy: DecisionPolicy,
    ) -> None:
        self._contract = contract
        self._registry = registry
        self._policy = policy

    def evaluate(self, context: DecisionContext) -> DecisionRecord:
        evaluations: list[GuaranteeEvaluation] = []

        for guarantee in self._contract.guarantees:
            evaluator = self._registry.get(guarantee.evaluator_id)
            result = evaluator.evaluate(output=context.output)

            evaluations.append(
                GuaranteeEvaluation(
                    guarantee_id=guarantee.guarantee_id,
                    passed=result.passed,
                    severity=guarantee.severity,
                    reason=result.reason,
                    evaluator_id=evaluator.evaluator_id,
                    evaluator_version=evaluator.version,
                )
            )

        outcome, reasons = self._policy.decide(guarantees=evaluations)

        return DecisionRecord(
            decision_id=str(uuid.uuid4()),
            context=context,
            guarantees=tuple(evaluations),
            outcome=outcome,
            reasons=reasons,
            created_at=datetime.now(timezone.utc),
        )