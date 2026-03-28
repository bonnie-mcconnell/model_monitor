## Behavioral contracts (this branch)

Classical ML monitoring catches statistical drift and performance degradation.
It doesn't catch the failure modes that matter most for language models: tone
shifting between versions, structured output silently breaking after a
fine-tune, safety posture eroding, instruction adherence degrading. These
failures don't show up in accuracy or latency metrics.

This branch adds behavioral contracts: explicit, versioned, enforceable
guarantees about how a model must behave in production.

A contract is a YAML file listing guarantees and the evaluator that checks
each one:
```yaml
contract_id: support_response
version: "1.0"
scope: chat_completion

guarantees:
  - id: valid_json
    description: Response must be valid JSON
    severity: CRITICAL
    evaluator: json_validity

  - id: response_schema_v1
    description: Response must conform to SupportResponse schema v1
    severity: CRITICAL
    evaluator: json_schema_v1
```

The `BehavioralContractRunner` evaluates each model output against every
guarantee, produces severity-scored results, and passes them to a
`DecisionPolicy`. The `StrictBehaviorPolicy` blocks on any CRITICAL failure
and warns on two or more HIGH failures. Severity uses explicit equality
comparison rather than `>=` because Python `Enum` doesn't support ordering
by default - using `>=` would silently pass on any severity level.

Every decision is recorded as an immutable `DecisionRecord` with full
provenance: which evaluator ran, which version, what the output was, what
the outcome was. This makes behavioral regressions auditable and replayable
- you can diff two `DecisionRecord`s from consecutive model versions to see
exactly what changed.

Two evaluators are implemented: `JsonValidityEvaluator` checks that output
parses as JSON. `JsonSchemaEvaluator` validates output against a JSON Schema
bound at construction time - one evaluator instance per schema version, which
makes the registry append-only rather than mutable. The schema is validated
at construction time so a malformed schema fails immediately rather than
silently passing every evaluation.

**What I'd do differently.** The `trust_score` in the main monitoring layer
doesn't yet incorporate behavioral violation rates. There's a TODO in
`trust_score.py` for a `behavioral_penalty` component. Wiring that in would
mean a model that's passing accuracy metrics but failing behavioral contracts
gets a lower trust score and triggers the policy engine, which is the actual
goal.

**What's next.** An LLM-as-judge evaluator that calls the Anthropic API to
assess tone consistency and instruction adherence across model versions. That
evaluator would take two outputs (one from the reference model, one from the
candidate) and return a structured comparison. Combined with the contract
system already here, it would give a full behavioral regression test suite
that runs automatically on every promotion.