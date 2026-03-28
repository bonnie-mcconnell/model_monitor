# model-monitor

ML monitoring system for production models. Detects feature drift using PSI,
tracks performance degradation across rolling windows, and triggers automated
lifecycle actions - retraining, rollback, promotion - through a policy engine
that is deliberately kept free of I/O and side effects.

## Why I built this

I was building out my ML portfolio and kept running into the same gap: most
tutorials show you how to train a model, but nothing shows you what happens
after it's deployed. Production models degrade. Features drift. The model you
shipped six months ago is quietly getting worse and nobody notices until
something breaks. I wanted to build the system that catches that - and to
understand the engineering trade-offs that make it actually work in production.

## What's hard about it

**PSI drift detection.** Population Stability Index requires reusing the
reference distribution's percentile bin edges when binning incoming data.
Using fixed-width bins produces misleading scores when the reference
distribution is skewed. The reference stats are computed once at training time
and stored in `data/reference/reference_stats.json`.

**The async/sync boundary.** The aggregation loop calls
`asyncio.create_task()` to schedule execution without blocking monitoring.
Putting that call inside a plain `def` was a bug in the original design -
`create_task` requires a running event loop and raises `RuntimeError` in
some contexts without failing loudly. Making `aggregate_once` properly
`async` was the correct fix, not wrapping it.

**Retrain idempotency.** If the system crashes mid-retrain and restarts,
it shouldn't kick off a duplicate retrain for the same evidence window.
The fix is a SHA-256 fingerprint of the evidence DataFrame, which is deterministic
across restarts, collision-resistant, and doesn't require a separate
deduplication store.

**Baseline wiring.** The decision engine needs to compare current F1 against
the deployed model's quality at promotion time, not against a rolling average
of recent batches. I shipped a version where `f1_baseline=summary.avg_f1`,
which made `f1_drop` always zero and meant retrain could never trigger from
the aggregation path. Finding that bug and tracing it to the right fix
- storing `baseline_f1` in `active.json` at promotion time - is the clearest
example of a design decision I had to reason through rather than just write.

## Architecture
```
aggregation → trust score → decision engine → executor
```

The **monitoring layer** records batch-level metrics and aggregates them into
rolling windows (5m, 1h, 24h). It emits signals only; no decisions are made
here. This separation means monitoring can't accidentally trigger actions.

The **trust score** is a weighted combination of accuracy, F1, confidence,
drift severity, and latency, bounded to [0,1]. It gives the decision engine
a single operational signal rather than requiring it to reason about five
independent numbers.

The **decision engine** is pure policy: no I/O, no persistence, no async
code. It receives metrics and returns an immutable `Decision` dataclass. The
priority order is: severe drift → reject, catastrophic F1 drop → rollback,
sustained degradation → retrain (with cooldown), N stable batches → promote.
Being a pure function makes it trivially testable and replayable from any
stored state.

The **executor** handles all side effects asynchronously. It enforces retrain
cooldowns, checks the idempotency key before acting, holds a lock to prevent
concurrent retrains, and supports `dry_run` for testing.

## Quick start
```bash
pip install -e ".[dev]"
make sim        # run the drift simulation loop
make run        # start the FastAPI server at localhost:8000
make test       # run the full test suite
```

## Key design decisions

**PSI not KS test.** PSI is interpretable: below 0.1 is stable, above 0.2
is severe, and these thresholds are configurable in `config/drift.yaml`.
KS gives a p-value, which is harder to threshold deterministically in a
policy engine. PSI also handles multivariate drift by averaging per-feature
scores.

**File-based model store.** Atomic rename - write to `.tmp`, then
`Path.replace()` - gives crash safety without a database. The trade-off is
that it doesn't work across multiple processes without a distributed lock,
which is fine for a single-node deployment but would need redesigning for
anything that scales horizontally.

**Baseline F1 at promotion time.** The baseline is written to `active.json`
when a model is promoted and read once per aggregation pass. This means the
decision engine always compares against what the deployed model was actually
measured at, not what it's doing today.

**Decision engine has no I/O.** All state the engine needs is passed in as
arguments. This makes decisions replayable: given the same inputs, you always
get the same decision. It also means the engine can be tested without any
mocking of databases or file systems.

## What I'd do differently

Crash recovery across restarts is incomplete. If the system dies mid-retrain,
the snapshot's `retrain_key` is lost and the idempotency check can't run on
restart. A proper fix would persist snapshots to the database before
execution, not after.

The dashboard currently has no endpoint that writes `MetricRecord` rows from
external inference - it only reads. Connecting a real inference pipeline
would require an ingest endpoint and authentication. That's the obvious next
step to make this useful beyond the simulation.