# Model Monitor — Architecture Overview

## Design Goals
- Deterministic decision-making
- Explainable automation
- Async-safe execution
- Clear separation of policy vs execution

## Core Concepts

### Monitoring
Batch-level metrics are recorded and aggregated into rolling windows.
No decisions are made at this stage.

### Trust Score
A bounded [0,1] trust score combines accuracy, F1, confidence, drift, and latency.
This provides a single operational signal for alerting and automation.

### Decision Engine
Pure policy layer.
Consumes metrics and trust signals and produces an immutable Decision.
Contains no I/O and no side effects.

### Decision Executor
Async-only side-effect layer.
Executes retraining, promotion, and rollback without blocking monitoring.

### Model Store
Atomic model lifecycle management:
- promotion
- rollback
- archival
File-based by default, abstracted behind a stable API.

## Execution Contexts
The system does not rely on a single global orchestrator.
Instead, each execution context (inference, simulation, background monitoring)
composes the same primitives:

aggregation → decision → execution

This avoids duplication while preserving clarity.

## Failure Handling
- Cold-start safe
- Retrain cooldown enforced
- Rollback only when archive exists
- No blocking operations in monitoring paths


# TODO, crash recovery across restarts