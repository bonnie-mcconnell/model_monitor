# model_monitor - Architecture

## Design goals

- Deterministic, replayable decisions
- Explainable automation: every decision includes a reason and metadata
- Clean separation of policy from execution
- Auditability: append-only decision log, immutable records throughout
- Composable: each runtime context composes the same primitives differently

---

## Classical monitoring pipeline

```
inference → MetricsStore → aggregation loop → trust score → decision engine → executor
```

### Monitoring layer
Records batch-level `MetricRecord`s to SQLite. Aggregates them into rolling
windows (5m, 1h, 24h). Emits signals only - no decisions are made here.
This separation means the monitoring layer cannot accidentally trigger actions.

### Trust score
A bounded [0, 1] score derived from six components:

| Component | Weight | Source |
|---|---|---|
| Accuracy | 30% (scaled) | batch accuracy_score |
| F1 | 25% (scaled) | batch f1_score |
| Confidence | 15% (scaled) | mean max class probability |
| Drift | 20% (scaled) | PSI converted to [0,1] |
| Latency | 10% (scaled) | decision time in ms |
| Behavioral | 15% (additive) | contract violation rate |

The behavioral component enters via `behavioral_violation_rate`: the proportion
of recent contract evaluations that resulted in BLOCK or WARN. Zero violations
contribute full score; 100% violation rate contributes zero. The remaining
five components are scaled down proportionally to accommodate the behavioral
weight, preserving their relative importance.

### Decision engine
Pure policy layer. No I/O, no persistence, no async code. Receives metrics
and returns an immutable `Decision`. Priority order:

1. Severe drift (PSI ≥ threshold) → **reject**
2. Catastrophic F1 drop (≥ max_f1_drop) → **rollback**
3. Sustained degradation (≥ min_f1_gain) → **retrain** (with cooldown)
4. N consecutive stable batches → **promote**
5. Default → **none**

Behavioral signals enter through the trust score - no separate code path needed.

### Decision executor
Async-only execution layer. Enforces retrain cooldowns, checks SHA-256
idempotency key before acting, holds an asyncio.Lock to prevent concurrent
retrains, supports dry_run for testing. All side effects are isolated here.

### Model store
File-based, crash-safe via atomic rename (`os.replace`). Supports promotion,
rollback, and archiving. Baseline F1 is written to `active.json` at promotion
time and read once per aggregation pass - never inferred from rolling averages.

---

## Behavioral contracts pipeline

```
LLM output → DecisionContext → BehavioralContractRunner → DecisionRecord
                                        ↑
                               EvaluatorRegistry + Contract (from YAML)
```

Runs independently of the classical pipeline. Produces `DecisionRecord`s
that feed `behavioral_violation_rate` back into the trust score.

### Contract
A versioned YAML file listing guarantees. Each guarantee specifies an
evaluator ID and a severity (CRITICAL, HIGH, LOW). Loaded at startup;
never mutated at runtime.

### EvaluatorRegistry
Append-only map of evaluator ID → evaluator instance. Duplicate registration
raises immediately. Unknown evaluator ID raises at evaluation time, not silently
passes. Evaluators satisfy the `GuaranteeEvaluator` Protocol - no inheritance
required.

### Evaluators (implemented)
| Evaluator | Type | What it checks |
|---|---|---|
| `JsonValidityEvaluator` | Structural | Output parses as JSON |
| `JsonSchemaEvaluator` | Structural | Output conforms to a JSON Schema (bound at construction) |
| `ToneConsistencyEvaluator` | Semantic | Cosine similarity between output embedding and centroid of reference embeddings ≥ threshold |
| `LLMJudgeEvaluator` | Semantic (LLM-as-judge) | Structured verdict from an injected `LLMClient`; `MockLLMClient` in tests, `AnthropicLLMClient` in production |

Both semantic evaluators inject their external dependency via a Protocol:
`ToneConsistencyEvaluator` takes a `TextEncoder`, `LLMJudgeEvaluator` takes
an `LLMClient`. Neither imports its real dependency at module level. This
means tests run in milliseconds with no model downloads and no API keys,
and production clients are swappable without touching the evaluator code.

### BehavioralContractRunner
For each guarantee: looks up evaluator, calls `evaluate(output=...)`, wraps
result in a `GuaranteeEvaluation` (with provenance: evaluator ID, version,
severity). Passes all evaluations to the policy. Returns an immutable
`DecisionRecord` with a UUID, full context, and UTC timestamp.

### StrictBehaviorPolicy
- Any CRITICAL failure → BLOCK
- Two or more HIGH failures → WARN
- Otherwise → ACCEPT

Uses explicit equality (`severity == Severity.CRITICAL`) rather than `>=`
because Python Enum does not define ordering by default.

### DecisionRecord
Immutable (`frozen=True`, `slots=True`). Contains full provenance: which
evaluator ran, which version, what the output was, what the outcome was.
`diff_decisions` compares two consecutive records to surface behavioral
regressions between model versions.

---

## Behavioral decision store

`BehavioralDecisionStore` persists one row per `BehavioralContractRunner`
evaluation to the `behavioral_evaluations` SQLite table. It provides
`violation_rate(since_ts)` - the proportion of evaluations in the given
time window that resulted in BLOCK or WARN.

The aggregation loop queries this rate over the same window as the
performance metrics, so both signals share a consistent denominator.
`behavioral_violation_rate` is then passed to `compute_trust_score`,
closing the loop between the two monitoring systems.

Writes are idempotent: duplicate `decision_id`s are silently skipped,
preventing double-counting if the caller retries on failure.

---

## Ingest API

`POST /metrics/ingest` accepts batch results from external inference pipelines.
Authenticated via `X-API-Key` header checked against `MONITOR_API_KEY`
environment variable. Returns 503 if the variable is not set (endpoint
administratively disabled). Returns 401 for wrong key. Returns 422 for
malformed payload (handled automatically by Pydantic).

---

## Failure handling

- Cold-start safe: no model → `can_decide=False`, no action taken
- Retrain cooldown: enforced by both batch index (ephemeral) and action history (durable)
- Rollback only when archive version exists
- No blocking operations in monitoring paths
- Explicit failure states in DecisionSnapshot (`pending/executed/skipped/failed`)
- Executor failures recorded to DecisionStore before re-raising

## Known limitations

- **Crash recovery**: retrains use `SnapshotStore` as a write-ahead log.
  Non-retrain actions (promote, rollback) have no snapshot persistence but
  are idempotent at the model-store level and safe to replay.
- **Model store is single-node**: `os.replace` is atomic within a filesystem
  but not across processes without a distributed lock. Horizontal scaling
  requires a different storage backend (object store + advisory lock).
- **Thresholds are hand-tuned**: trust score weights and the behavioral
  budget (50ms) are reasoned defaults, not calibrated from production data.
  A proper calibration would use historical latency distributions and
  observed violation rates from a real deployment.


## Bugs found and fixed by the test suite

**Floating point promotion threshold** (`training/promotion.py`): `0.82 - 0.80`
evaluates to `0.019999...` in IEEE 754 - less than `0.02`. Without an epsilon
tolerance, a candidate whose F1 improves by exactly `min_improvement` would
be silently rejected. Fixed with `_IMPROVEMENT_EPS = 1e-9`.

**Entropy non-negativity** (`utils/stats.py`): additive EPS smoothing
(`probs + 1e-9`) in `entropy_from_labels` caused a tiny negative result
(`-1e-9`) for pure distributions. Shannon entropy is non-negative by
definition. Fixed with `max(0.0, ...)`.

**ORM state leaked into API responses** (`api/dashboard.py`): using `__dict__`
on SQLAlchemy ORM rows includes `_sa_instance_state` in the response dict.
FastAPI silently drops non-serialisable keys in the current version, but the
behaviour is implementation-dependent and exposes internals. Fixed with
`_orm_to_dict()` which reads only mapped column values.

**Decision metadata not persisted** (`storage/decision_store.py`): `Decision.metadata`
contained the context that produced a decision (baseline F1, threshold at the time,
cooldown state) but was silently dropped on write. If a reject or rollback fired and you
later wanted to know what threshold was in effect, the audit log could not tell you.
Fixed by adding a `metadata_json` TEXT column to `decision_history` and serialising
`Decision.metadata` on every `record()` call. The field is `None` for decisions with
empty metadata, matching the pre-existing convention for optional audit fields.

**In-sample evaluation in `RetrainPipeline`** (`training/retrain_pipeline.py`):
candidate F1 was measured on the same data the model was trained on, producing
optimistic in-sample estimates that favour promotion of overfit models. Fixed by
holding out 20% of retrain data for evaluation. Both candidate and current model
are evaluated on the same held-out set for a fair comparison.
