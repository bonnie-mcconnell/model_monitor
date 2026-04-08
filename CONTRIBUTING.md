# Contributing to model-monitor

This document describes the standards that apply to all code in this repo.
They exist because a monitoring system that silently produces wrong results
is worse than no monitoring system at all. Every contribution must keep
the system trustworthy.

---

## Setup

```bash
git clone https://github.com/bonnie-mcconnell/model_monitor
cd model_monitor
pip install -e ".[dev]"
make test        # 306 tests should pass on behavior-monitoring, 76 on main
make lint        # ruff must be clean
make typecheck   # mypy must be clean
```

If any of these fail on a clean clone, that is a bug in the repo.

---

## Standards that apply to everything

**Every function has full type annotations.** No bare `def f(x):` - always
`def f(x: float) -> str:`. This is enforced by mypy and CI will reject
unannotated public functions.

**No bare excepts.** `except Exception as exc:` minimum, with the exception
used in the handler. Silent failures are worse than crashes in a monitoring
system.

**No silent fallbacks.** If a function cannot succeed, it must raise. Returning
`None` or a default when an error occurred creates invisible failures that
produce misleading monitoring signals.

**Comments explain why, not what.** `# sort by timestamp` describes what the
next line does and adds no information. `# oldest-first so the cursor boundary
is stable under concurrent inserts` explains a non-obvious choice.

**Tests test edge cases, not just happy paths.** A test that only verifies
the function runs without crashing is not much better than no test. Every
test should name the property it is verifying, ideally in its docstring.

---

## Adding a new evaluator

Evaluators are the extension point for the behavioral contracts system.
To add one:

1. Create a class in `src/model_monitor/contracts/behavioral/evaluators.py`
   that satisfies the `GuaranteeEvaluator` Protocol:
   ```python
   class MyEvaluator:
       evaluator_id = "my_evaluator"   # unique, stable across versions
       version = "1.0"

       def evaluate(self, *, output: str) -> EvaluationResult:
           ...
   ```

2. If the evaluator has an external dependency (a model, an API client),
   inject it via a Protocol rather than constructing it internally. See
   `ToneConsistencyEvaluator` (injected `TextEncoder`) and
   `LLMJudgeEvaluator` (injected `LLMClient`) for the pattern.

3. Write at least four tests:
   - Passes when output meets the guarantee
   - Fails when output violates the guarantee
   - Correct reason string format on failure
   - At least one edge case (empty input, single reference, boundary value)

4. Verify `isinstance(evaluator, GuaranteeEvaluator)` passes - this is
   checked by `validate_contract` at startup.

---

## Adding a decision rule to the engine

`DecisionEngine.decide()` evaluates rules in priority order. Before adding
a rule:

- Confirm no existing rule covers the same condition
- Place it at the correct priority level (drift before regression before
  degradation before promotion)
- Write a test that confirms it fires at the exact threshold, not just
  "above" or "below" it - boundary conditions are where bugs live
- Write a test that confirms the cooldown or hysteresis logic is respected

---

## Changing the trust score formula

The trust score is a contract: downstream systems (the decision engine,
dashboards, alerting) depend on it staying bounded to [0, 1] and on the
weights summing to 1.0. Before changing weights or adding components:

- Run `test_weights_sum_to_one_with_default_behavioral_weight` - this must
  still pass
- Run `test_perfect_inputs_always_score_one_regardless_of_behavioral_weight`
  - this verifies the scaling invariant
- Update `ARCHITECTURE.md` trust score table with new weights

---

## What CI checks

Every push and PR runs:

1. `ruff check src/ tests/` - no unused imports, no inline imports, no dead assignments
2. `mypy src/model_monitor/ tests/` - no type errors in source or tests
3. `pytest tests/ -v` - all tests pass

All three must pass. A green CI badge means all three are clean.

---

## What does not need tests

- ORM column definitions (`storage/models/*.py`) - these are schema declarations
- Pure dataclasses and Enums with no logic
- The Streamlit UI (`ui/streamlit_app.py`) - presentation layer, no logic
- The simulation script (`scripts/simulation_loop.py`) - integration harness

Everything else with branching logic should have tests.
