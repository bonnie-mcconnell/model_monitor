# Model Monitor - Architecture Overview
## Design goals

- Deterministic decision-making
- Explainable automation
- Async-safe execution
- Clear separation of policy vs execution
- Auditability and rollback safety

## Core components
### Monitoring layer

- Records batch-level metrics
- Aggregates metrics into rolling windows
- Computes drift and trust scores

No decisions are made here.
Monitoring emits signals only.

### Trust score

A bounded [0, 1] score derived from:
- accuracy
- F1
- confidence
- drift severity
- optional latency

This provides a single operational signal for:
- alerting
- policy evaluation
- dashboards

### Decision engine

- Pure policy layer
- Deterministic and side-effect free
- Consumes metrics and trust signals
- Produces an immutable Decision

The engine contains no async code, no I/O, and no persistence.

### Decision executor

- Async-only execution layer
- Executes retraining, promotion, and rollback
- Enforces:
- cooldowns
- evidence thresholds
- idempotency
- Supports dry-run execution

All side effects are isolated here.

### Model store

Responsible for:
- model persistence
- promotion
- archival
- rollback safety

Key properties:
- atomic promotion
- no in-place overwrites
- append-only metadata

File-based by default, abstracted behind a stable interface.

### Execution model

The system avoids a single global orchestrator.

Instead, each runtime context (simulation, inference, background jobs) composes:

aggregate → decide → execute

This minimizes duplication while preserving clarity and testability.

### Failure handling

- cold-start safe
- retrain cooldown enforced
- rollback only when archive exists
- no blocking operations in monitoring paths
- explicit failure states in decision snapshots

Failures are surfaced early and explicitly.

## Known limitations

- Crash recovery for retrains is implemented via the `DecisionSnapshot` write-ahead
  pattern. Non-retrain actions (promote, rollback) are idempotent at the model-store
  level and do not require snapshot persistence.
- Model store is single-node: `os.replace` is atomic within a filesystem but not
  across processes without a distributed lock. Horizontal scaling requires a
  different storage backend.
- Thresholds are hand-tuned. The trust score weights and the retrain/rollback
  thresholds are reasoned defaults, not calibrated from historical deployment data.

## Bugs found and fixed by the test suite

**Floating point promotion threshold** (`training/promotion.py`): `0.82 - 0.80`
evaluates to `0.019999...` in IEEE 754 - less than `0.02`. Without an epsilon
tolerance, a candidate whose F1 improves by exactly `min_improvement` would be
silently rejected. Fixed with `_IMPROVEMENT_EPS = 1e-9`.

**Entropy non-negativity** (`utils/stats.py`): additive EPS smoothing in
`entropy_from_labels` caused a tiny negative result for pure distributions.
Shannon entropy is non-negative by definition. Fixed with `max(0.0, ...)`.

**ORM state leaked into API responses** (`api/dashboard.py`): using `__dict__`
on SQLAlchemy ORM rows includes `_sa_instance_state`. Fixed with `_orm_to_dict()`
which reads only mapped column values.


**Decision metadata not persisted** (`storage/decision_store.py`): `Decision.metadata`
contained the context that produced a decision (baseline F1, threshold at the time,
cooldown state) but was silently dropped on write. Fixed by adding a `metadata_json`
TEXT column to `decision_history` and serialising `Decision.metadata` on every
`record()` call.
**In-sample evaluation in `RetrainPipeline`** (`training/retrain_pipeline.py`):
candidate F1 was measured on training data, producing optimistic estimates that
favour overfit models. Fixed with a 20% held-out validation split.
