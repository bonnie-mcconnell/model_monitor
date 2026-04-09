# model_monitor - Architecture

## Design goals

- **Deterministic, replayable decisions.** Given the same inputs, the decision
  engine always produces the same output. Every decision is reconstructible
  from the audit log.
- **Clean separation of policy from execution.** The engine returns an immutable
  `Decision`; the executor handles all side effects. Neither knows about the
  other's internals.
- **Fail loudly.** A monitoring system that swallows errors produces false
  confidence. Every component raises on invalid input rather than returning
  a default that silently propagates.
- **Auditability.** Every decision is persisted with its full context
  (`metadata_json`) so the reason for a rollback or retrain is recoverable
  months later without reconstructing state from other tables.

---

## Pipeline overview

```
predict_batch → MetricsStore → aggregation loop → trust score → decision engine → executor
                                                                                      │
                                                                              model store
                                                                              DecisionStore
```

Each stage has one responsibility and no knowledge of its neighbours'
internals. The monitoring layer cannot accidentally trigger actions; the
decision engine cannot accidentally persist state.

---

## Component reference

### MetricsStore

SQLite-backed persistence for batch-level `MetricRecord`s.

- Cursor-based pagination: `list(cursor=...)` is stable under concurrent
  writes because it pages by `(timestamp, id)` rather than `OFFSET`.
- `write()` rolls back on any exception - no partial records.

### Aggregation loop

Async background task that runs once per `poll_interval` (default 60s).

Aggregates `MetricRecord`s into three rolling windows:

| Window | Seconds |
|--------|---------|
| `5m`   | 300     |
| `1h`   | 3 600   |
| `24h`  | 86 400  |

For each window it: reads records, computes the trust score, persists the
summary, calls the decision engine, and schedules execution via
`_schedule_execution`. The `_schedule_execution` helper attaches
`add_done_callback` to the background task so executor failures are logged
rather than silently discarded at GC time.

### Trust score

A bounded [0, 1] weighted sum of five performance components:

| Component | Weight | Source |
|-----------|--------|--------|
| Accuracy  | 30%    | `accuracy_score(y_true, preds)` |
| F1        | 25%    | `f1_score(y_true, preds)` |
| Confidence | 15%   | mean max class probability |
| Drift     | 20%    | PSI converted to [0, 1] |
| Latency   | 10%    | end-to-end decision time in ms |

Drift-to-trust mapping: PSI < 0.1 → 1.0, PSI > 0.3 → 0.0, linear between.
Latency-to-trust mapping: ≤ 300ms → 1.0, ≥ 1500ms → 0.0, linear between.

### Decision engine

Pure policy layer. No I/O, no persistence, no async code.

Priority order (evaluated top to bottom; first match wins):

1. `drift_score ≥ psi_threshold` → **reject**
2. `f1_drop ≥ max_f1_drop` → **rollback**
3. `f1_drop ≥ min_f1_gain` → **retrain** (subject to cooldown)
4. Last N actions all `none` → **promote**
5. Default → **none**

**Dual cooldown for retrain:**
- *Ephemeral*: `_last_retrain_batch` on the engine instance; suppresses
  retrains within `cooldown_batches` of the last one. Reset on process
  restart.
- *Durable*: checks the last N actions in `DecisionStore.tail()`; survives
  restarts. Both must clear before retrain fires again.

The `f1_baseline` used for comparison is written to `active.json` at
promotion time and never updated. A rolling baseline would drift with the
model, making `f1_drop` approach zero even as absolute performance collapses.

### Decision executor

Async orchestration layer. All model lifecycle side effects live here.

Responsibilities:
- Hold an `asyncio.Lock` so concurrent retrains cannot fire simultaneously
- Compute a SHA-256 fingerprint of the retrain evidence DataFrame before
  executing; skip if the key is already known to `SnapshotStore`
- Write the snapshot to `SnapshotStore` *before* execution begins
  (write-ahead log pattern - see Crash safety below)
- Support `dry_run=True` for testing without real side effects

### Model store

File-based, crash-safe via `os.replace` (atomic within a filesystem).

Layout:
```
models/
  current.pkl          ← active model
  candidate.pkl        ← candidate awaiting promotion decision
  archive/
    model_<version>.pkl
  active.json          ← version + baseline_f1 + promotion timestamp
```

Lazy singleton at module level: `ModelStore()` is not called at import
time, preventing the side effect of creating `models/` and `models/archive/`
in whatever directory the process happens to be in when the module loads.

### RetrainPipeline

Trains a candidate model and evaluates it on a held-out split.

- **20% validation split** with `random_state=42` for reproducibility.
  Both candidate and current model are evaluated on the *same* held-out
  set - not on the data the candidate was trained on.
- Falls back to full dataset when `n < 50` rows (a reliable 20% split
  requires at least 10 validation rows).
- `min_f1_improvement` uses `_IMPROVEMENT_EPS = 1e-9` to handle
  IEEE 754 rounding: `0.82 - 0.80 = 0.019999...` without the epsilon.

---

## Storage schema

### `metrics_records`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | |
| `timestamp` | FLOAT | Unix epoch; indexed |
| `batch_id` | TEXT | caller-assigned |
| `n_samples` | INTEGER | |
| `accuracy` | FLOAT | [0, 1] |
| `f1` | FLOAT | [0, 1] |
| `avg_confidence` | FLOAT | [0, 1] |
| `drift_score` | FLOAT | PSI value |
| `decision_latency_ms` | FLOAT | |
| `action` | TEXT | DecisionType |
| `reason` | TEXT | |
| `previous_model` | TEXT | nullable |
| `new_model` | TEXT | nullable |

### `decision_history`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | |
| `timestamp` | FLOAT | indexed |
| `batch_index` | INTEGER | nullable |
| `action` | TEXT | |
| `reason` | TEXT | |
| `trust_score` | FLOAT | nullable |
| `f1` | FLOAT | nullable |
| `drift_score` | FLOAT | nullable |
| `model_version` | TEXT | nullable |
| `metadata_json` | TEXT | full Decision.metadata as JSON |

### `metrics_summary`

One row per aggregation window (upserted each pass).

### `metrics_summary_history`

Append-only; one row per window per aggregation run. Used for trend charts.

---

## Crash safety

**Retrain deduplication** (`SnapshotStore`):

```
executor                          SnapshotStore
   │                                   │
   │── compute retrain_key ────────────│
   │── snapshot_store.write() ─────────│── INSERT (pending)
   │                                   │
   │── execute retrain ─────── [crash here → key is in DB]
   │                                   │
   │── snapshot_store.update_status()──│── UPDATE (executed)
```

On restart: `is_retrain_key_known(key)` returns `True` → skip duplicate.

Non-retrain actions (promote, rollback) are idempotent at the model-store
level via `os.replace` and do not require snapshot persistence.

---

## Known limitations

- **Model store is single-node.** `os.replace` is atomic within a filesystem
  but not across processes without a distributed lock. Horizontal scaling
  requires a different storage backend (object store + advisory lock).
- **Thresholds are hand-tuned.** Trust score weights and PSI thresholds are
  reasoned defaults, not calibrated from production deployment data.
- **No ingest API on this branch.** `POST /metrics/ingest` with API-key
  authentication is implemented in `behavior-monitoring`.

---

## Bugs found and fixed by the test suite

**Floating point promotion threshold** (`training/promotion.py`):
`0.82 - 0.80` evaluates to `0.019999...` in IEEE 754 - less than `0.02`.
Without an epsilon tolerance, a candidate whose F1 improves by exactly
`min_improvement` would be silently rejected. Fixed with `_IMPROVEMENT_EPS = 1e-9`.

**Entropy non-negativity** (`utils/stats.py`): additive EPS smoothing in
`entropy_from_labels` caused a tiny negative result (`-1e-9`) for pure
distributions. Shannon entropy is non-negative by definition.
Fixed with `max(0.0, ...)`.

**ORM state leaked into API responses** (`api/dashboard.py`): `__dict__`
on SQLAlchemy ORM rows includes `_sa_instance_state`. Fixed with
`_orm_to_dict()` which reads only mapped column values.

**In-sample evaluation** (`training/retrain_pipeline.py`): candidate F1
was measured on training data, producing optimistic estimates that favour
overfit models. Fixed with a 20% held-out validation split.

**Decision metadata silently dropped** (`storage/decision_store.py`):
`Decision.metadata` contained the context that produced a decision
(baseline F1, threshold, cooldown state) but was not persisted. Fixed by
adding `metadata_json TEXT` and serialising on every `record()` call.

**Cold-start trust score race** (`core/decision_runner.py`):
`MetricsSummaryORM.trust_score` defaults to `0.0` before the aggregation
loop has run. Without a guard, `DecisionRunner.run_once()` would pass
`trust_score=0.0` to the engine and potentially trigger a spurious retrain
on first startup. Fixed with `if summary.n_batches == 0: continue`.
