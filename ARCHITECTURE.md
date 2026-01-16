# Model Monitor — Architecture Overview
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