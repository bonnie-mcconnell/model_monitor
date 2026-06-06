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
make test        # 806 tests should pass on main, 995 on behavior-monitoring
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
produce misleading monitoring signals. The one deliberate exception is
`DefaultModelActionExecutor._refresh_reference_stats()`, which is best-effort
and logs failures instead of raising - a stale reference file is preferable
to blocking a successful promotion.

**Comments explain why, not what.** `# sort by timestamp` describes what the
next line does and adds no information. `# oldest-first so the cursor boundary
is stable under concurrent inserts` explains a non-obvious choice.

**Tests test edge cases, not just happy paths.** A test that only verifies
the function runs without crashing is not much better than no test. Every
test should name the property it is verifying, ideally in its docstring.

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

- Change `config/trust_score.yaml` - never the source code directly
- Verify the five weights still sum to 1.0 (the Pydantic validator will
  catch this at startup if they don't)
- Verify that perfect inputs (accuracy=1, f1=1, drift=0, latency=0ms)
  still produce a trust score of 1.0
- Update the weight table in `ARCHITECTURE.md` and `README.md`

---

## Changing alerting thresholds

`MIN_TRUST_SCORE` and `CRITICAL_TRUST_SCORE` in `monitoring/thresholds.py`
are the only constants in that file that are imported by production code
(by `alerting.py`). All other constants in `thresholds.py` are reference
documentation - they reflect the YAML config files but are not used at
runtime. If you change a YAML threshold value, update the corresponding
reference constant in `thresholds.py` to keep the documentation accurate.

---

## Configuring policy thresholds

All decision engine thresholds live in YAML files under `src/model_monitor/config/`:

| File                | What it controls                              |
|---------------------|-----------------------------------------------|
| `drift.yaml`        | PSI threshold, drift window                   |
| `retrain.yaml`      | F1 gain required, cooldown, sample count      |
| `rollback.yaml`     | F1 drop that triggers immediate rollback      |
| `trust_score.yaml`  | Component weights (must sum to 1.0)           |

Do not hard-code threshold values in source code. `load_config()` is the
single point of truth for the live system's policy parameters.

---

## What CI checks

Every push and PR runs, on Python 3.10, 3.11, and 3.12:

1. `ruff check src/ tests/` - no unused imports, no inline imports, no dead assignments
2. `mypy src/model_monitor/ tests/` - no type errors in source or tests
3. `pytest tests/ -v --cov=model_monitor --cov-fail-under=80` - all tests pass
   with at least 80% coverage on the testable source

All three must pass. A green CI badge means all three are clean on all three
Python versions.

---

## What does not need tests

- ORM column definitions (`storage/models/*.py`) - these are schema declarations
- Pure dataclasses and Enums with no logic
- The Streamlit UI (`ui/streamlit_app.py`) - presentation layer, no logic
- The simulation script (`scripts/simulation_loop.py`) - integration harness

Everything else with branching logic should have tests. The API layer is
covered by `tests/test_api_routes.py`, `tests/test_dashboard_auth.py`,
`tests/test_simulate_endpoint.py`, and `tests/test_production_readiness.py`.
