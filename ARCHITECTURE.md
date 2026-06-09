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
predict_batch → RawDataBuffer ─────────────────────────────────────────────────────┐
     │                                                                              │
     └──→ MetricsStore → aggregation loop → trust score → decision engine → executor
                                │                                                   │
                           check_alerts                          ┌──────────────────┘
                           WebhookAlerter                        │
                                                    ┌────────────┼────────────────┐
                                               model store  DecisionStore  RawDataBuffer
```

Each stage has one responsibility and no knowledge of its neighbours'
internals. The monitoring layer cannot accidentally trigger actions; the
decision engine cannot accidentally persist state.

Two buffers feed the executor's retrain path:

- **`RetrainEvidenceBuffer`** accumulates aggregated monitoring signals to
  determine *when* to retrain. Lightweight; survives checkpointing.
- **`RawDataBuffer`** accumulates the actual labeled `(X, y)` pairs from
  inference batches and provides *what to train on*. FIFO-capped at 50k rows.
  Without this buffer, the executor had no valid training data - it was passing
  monitoring-aggregate rows to `train_model()`, which requires feature columns
  and a `label` column and crashed with `ValueError` on every retrain.

---

## Component reference

### MetricsStore

SQLite-backed persistence for batch-level `MetricRecord`s.

- Cursor-based pagination: `list(cursor=...)` is stable under concurrent
  writes because it pages by `(timestamp, id)` rather than `OFFSET`.
- `write()` rolls back on any exception - no partial records.

### Aggregation loop

Async background task that runs once per `poll_interval` (default 60s).

Aggregates `MetricRecord`s into three rolling windows, defined in
`monitoring/windows.py` - the single source of truth for window constants:

| Window | Seconds |
|--------|---------|
| `5m`   | 300     |
| `1h`   | 3 600   |
| `24h`  | 86 400  |

For each window it: reads records, computes the trust score, persists the
summary, calls the decision engine, fires alerts, and schedules execution
via `_schedule_execution`. The `_schedule_execution` helper attaches
`add_done_callback` to the background task so executor failures are logged
rather than silently discarded at GC time.

### Trust score

A bounded [0, 1] weighted sum of **seven** performance components.

| Component    | Weight | Source                                          |
|--------------|--------|-------------------------------------------------|
| Accuracy     | 23%    | `accuracy_score(y_true, preds)`                 |
| F1           | 18%    | `f1_score(y_true, preds)`                       |
| Calibration  | 14%    | ECE - Expected Calibration Error → [0, 1]       |
| Drift        | 18%    | input PSI converted to [0, 1]                   |
| Latency      | 17%    | **p95** latency (falls back to mean); ms → [0,1]|
| Data quality |  5%    | null rate + range violations + schema check     |
| Behavioral   |  5%    | contract violation EMA (BM branch)              |

All seven weights are defined in `config/trust_score.yaml` and validated at
startup - misconfigured weights (not summing to 1.0 ± 1e-6) fail loudly with
an explicit error message showing the individual values.

Key changes from v1:

- **Calibration replaces raw confidence.** ECE is a strictly better measure -
  a model that always outputs 90% confidence on 70%-accurate predictions has
  high mean confidence but high ECE. Guo et al. (2017) threshold: ECE > 0.05
  maps to trust ≤ 0.5; ECE ≥ 0.10 maps to trust 0.0.

- **p95 latency replaces mean latency.** Average latency can look healthy
  while p99 is pathological. The trust score now penalises the 95th-percentile
  per-sample latency, computed on batches of ≥ 20 samples.

- **Data quality is a first-class component.** `DataQualityMonitor` aggregates
  null rate, out-of-range violations, and schema consistency into a single
  [0, 1] score that feeds directly into the trust formula.

Drift-to-trust: PSI < 0.1 → 1.0, PSI > 0.3 → 0.0, linear between.
Latency-to-trust (p95): ≤ 300ms → 1.0, ≥ 1500ms → 0.0, linear between.

### PSI drift detection

Population Stability Index is computed in `monitoring/drift.py`.

**Bin edges are computed once at training time** (in `training/train.py:
compute_reference_stats`) and stored in `data/reference/reference_stats.json`
under the key `psi_bin_edges` for each feature. At inference time,
`DriftMonitor` accepts these pre-computed edges via `stored_bin_edges` and
passes them to `compute_psi`, which uses them directly instead of recomputing
from the reference array.

This is the critical property that makes PSI a valid drift signal: the
reference and production histograms are built on the *same* bin boundaries.
If edges were recomputed from production data, the two distributions would be
measured on different scales, and the PSI value would reflect scale shifts as
much as genuine distributional drift.

`compute_psi` accepts an optional `bin_edges` parameter:

- **With `bin_edges`** (normal production path): uses pre-computed training
  edges. This is what `DriftMonitor.update()` uses when `stored_bin_edges`
  is supplied at construction.
- **Without `bin_edges`** (fallback path): derives edges from `expected` via
  `np.percentile`. Used when `reference_stats.json` was produced by an older
  `train.py` that did not store `psi_bin_edges`, preserving backward
  compatibility.

### RawDataBuffer

`monitoring/raw_data_buffer.py`. Accumulates labeled `(X, y)` pairs from
inference batches for use as training data when a retrain fires.

- **FIFO eviction**: when `max_rows` (default 50k) is exceeded, the oldest
  chunk is dropped before appending the new one.
- **Schema enforcement**: the first `add_batch` call establishes the feature
  schema; subsequent calls with different column names raise `ValueError`.
- **`consume()` returns a DataFrame** with columns `[*feature_names, "label"]`
  - the exact format expected by `train_model()`.
- Designed for single-threaded use within the asyncio event loop.

The executor resolves training data in priority order:
1. Explicit `retrain_df` in context (tests, simulation loop)
2. `RawDataBuffer` when wired and ready
3. Synthetic fallback via `make_dataset()` - logged at `WARNING`

### TrustScoreConfig

`config/settings.py`. Seven component weights loaded from `config/trust_score.yaml`.

A `@model_validator` enforces that weights sum to 1.0 ± 1e-6 at construction
time. Misconfigured deployments fail at startup with an explicit error that
shows the individual weight values and their total - not silently producing
trust scores that sum to more or less than 1.0.

`compute_trust_score()` accepts `config: TrustScoreConfig | None`. When `None`,
it uses the package defaults (0.23/0.18/0.14/0.18/0.17/0.05/0.05 for
accuracy/F1/calibration/drift/latency/data\_quality/behavioral), preserving
backward compatibility with code that constructs the function call directly.

### ShapDriftAttributor

`monitoring/shap_attribution.py`. Computes per-feature SHAP importance shift
relative to a training-time baseline.

**Why SHAP alongside PSI?** PSI tells you that feature `f3`'s distribution
shifted. It does not tell you whether the model *uses* `f3` heavily in the
region where it shifted. A feature can drift dramatically in a tail the model
ignores, while a small shift in a high-weight region has large prediction
impact. SHAP importance shift separates these cases.

The shift for feature `f` is:

```
shift_f = mean|SHAP_current_f| - mean|SHAP_baseline_f|
```

A positive shift means the model is relying on that feature more than at
training time. The baseline is computed once at construction from `reference_X`.

**Latency**: ~30ms for a 200-tree RandomForest on a 200-row batch. Not
suitable for per-request serving paths; designed for batch post-prediction use.

**shap API compatibility**: handles both shap < 0.40 (list of arrays) and
shap >= 0.40 (3-D array `(n_samples, n_features, n_classes)`) output formats.

### ShadowPredictor

`inference/shadow.py`. Runs a candidate model silently alongside the primary.

The primary's output is always returned. When a candidate is set:
1. `predict_batch` runs the primary and returns its output.
2. The candidate runs in the same call with `batch_id + "_shadow"`.
3. Agreement rate, candidate F1, and candidate trust score are accumulated in
   `ShadowStats`.
4. Candidate failures are caught and logged - the primary path is unaffected.

`ShadowStats.candidate_beats_primary` is true when both `mean_candidate_f1 >
mean_primary_f1` and `mean_candidate_trust > mean_primary_trust` over all
accumulated batches. This is a live-traffic promotion signal that complements
the static holdout evaluation in `RetrainPipeline`.

`consume_shadow_stats()` returns the accumulated stats and resets the counter -
the same consume-and-clear pattern as `RawDataBuffer`. `set_candidate()` also
resets stats so a new candidate starts with a clean slate.



### OutputDriftMonitor

`monitoring/output_drift.py`.  PSI applied to the model's predicted
probability vectors rather than to input features.  Output drift is often
detectable *before* performance metrics degrade: the prediction distribution
can shift (e.g. class imbalance, score compression) while batch F1 is still
within normal variance.

Reference bin edges are fixed from the training-time probability distribution,
exactly mirroring the `DriftMonitor` design.  The aggregate score is the mean
PSI across output classes; `last_class_scores` stores the per-class breakdown
for dashboard inspection.

### DataQualityMonitor

`monitoring/data_quality.py`.  Checks each inference batch for upstream data
pipeline failures before those failures corrupt metrics.

Three sub-checks, scored 0–1 and averaged into `quality_score`:
- **Null rate** - fraction of null cells; penalised above `max_null_rate` (5%)
- **Out-of-range** - fraction of configured features with values outside
  their physical bounds (e.g. age < 0, probability > 1.0)
- **Schema consistency** - missing or unexpected column names

The score feeds directly into `compute_trust_score()` as the `data_quality`
component.  Detects upstream data pipeline failures before they corrupt metrics.

### CausalDriftAttributor

`monitoring/causal_drift.py`. Distinguishes genuine distributional shift from
upstream pipeline failures using Granger causality tests.

SHAP measures correlation between features and output changes, but cannot
distinguish cause A (real population shift - retrain needed) from cause B
(pipeline bug - retrain would be harmful). The Granger test asks: "does the
past of feature X help predict the future of feature Y, beyond Y's own
history?" Features that drift without their Granger-parents drifting are
classified as ``pipeline_suspect``.

Output: ``CausalDriftReport`` with per-feature ``DriftClass`` and a
``dominant_cause`` summary (``genuine_shift``, ``pipeline_failure``,
``mixed``, or ``none``).

### ThresholdAdvisor

`monitoring/threshold_advisor.py`. Calibrates PSI and trust score thresholds
from stable reference observations.

Records stable-period PSI and trust scores; the calibrated warn threshold is
the ``(1-alpha)`` percentile of stable-period values. At alpha=0.05 only 5%
of stable batches exceed the threshold - the correct false-positive rate for
a one-sided test. Generates notes on high-variance features where the default
0.10 PSI threshold would cause frequent false positives.

### ConformalMonitor

`monitoring/conformal.py`.  Rigorous coverage monitoring using split conformal
prediction (LAC variant - Least Ambiguous set Classifier).

The three-layer monitoring stack:
1. PSI (input shift) - inputs are changing
2. Output PSI (output shift) - model outputs are changing
3. Conformal coverage (prediction quality) - model correctness guarantees
   are breaking down

Calibration sets the nonconformity threshold `q_hat` at the
`ceil((n+1)(1-alpha)/n)` quantile of `1 - P(true class)` on held-out data.
On production batches:
- **With labels**: `coverage_rate` is compared to the `1-alpha` guarantee.
  `coverage_ok = False` when empirical coverage falls more than 2 SE below
  the target.
- **Without labels**: `mean_set_size` is tracked as a coverage proxy.
  Growing set size indicates declining model confidence.

Reference: Angelopoulos & Bates (2021), arXiv:2107.07511.

### Decision engine

Pure policy layer. No I/O, no persistence, no async code.

Priority order (evaluated top to bottom; first match wins):

1. `drift_score ≥ psi_threshold` → **reject** (escalates to **retrain** on sustained rejection)
2. `f1_drop ≥ max_f1_drop` → **rollback**
3. `f1_drop ≥ min_f1_gain` → **retrain** (subject to dual cooldown + circuit breaker)
4. Last N actions all `none` → **promote**
5. Default → **none**

**Dual cooldown for retrain:**

- *Ephemeral*: `_last_retrain_batch` on the engine instance suppresses
  retrains within `cooldown_batches` of the last one. Resets on process restart.
- *Durable*: checks the last N actions in `DecisionStore.tail()`; survives restarts.

Both cooldowns operate independently. A retrain is blocked when *either* is active.
Tests in `test_decision_hysteresis.py` cover the disagreement case explicitly.

**Retrain circuit breaker** (`max_retrain_attempts` in `retrain.yaml`, default 10):

Without a cap, if every retrain produces a model that still drifts, the engine
retries indefinitely - burning compute and masking the underlying data problem.
When `_retrain_attempt_count ≥ max_retrain_attempts` the engine emits
`system_error` instead of `retrain`. Call `engine.reset_retrain_counter()` after
diagnosing and fixing the root cause. Both retrain paths (degradation and
reject-escalation) share the same counter and are both subject to the cap.
Set `max_retrain_attempts: 0` to disable the breaker entirely.

The `f1_baseline` used for comparison is written to `active.json` at
promotion time and never updated. A rolling baseline would drift with the
model, making `f1_drop` approach zero even as absolute performance collapses.

### Alerting

`monitoring/alerting.py` provides two independent alert paths:

**`check_alerts()`** - in-process log alerts. Fires structured log records
at `WARNING` or `ERROR` level when trust score crosses `MIN_TRUST_SCORE`
(0.70) or `CRITICAL_TRUST_SCORE` (0.60). Per-key cooldown suppression
prevents alert fatigue on persistent degradation. Thresholds are defined in
`monitoring/thresholds.py` and are independent of the decision engine's
policy thresholds (which come from the YAML config files).

When `alert_store` is supplied (wired in via `startup.py`), every fired alert
is persisted to `storage/alert_store.py:AlertStore`. Cooldown-suppressed
alerts are **not** persisted - only genuinely new alerts are stored. This
makes alert rate a first-class observable metric: `AlertStore.count_since(t)`
answers "how many critical alerts in the last hour?" in O(1).

**`WebhookAlerter`** - outbound HTTP notification. POSTs a JSON payload
`{"window", "trust_score", "severity", "ts"}` to a configurable URL
(Slack incoming webhook, PagerDuty Events API v2, etc.). Key properties:

- `timeout_s=2.0` prevents a slow endpoint from blocking the pipeline
- `severity_filter` suppresses low-priority alerts (e.g. critical-only paging)
- Network and HTTP errors are caught and logged, never re-raised
- `_post` parameter is injectable - tests never make real HTTP calls
- `requests` is imported lazily - zero import-time cost when no webhook is configured

### Prometheus metrics

`GET /metrics` (implemented in `api/metrics.py`) returns current monitoring
state using the official `prometheus-client` library. All metric types are
correct and interoperable with any Prometheus-compatible backend.

Exported metrics:

| Metric | Type | Labels | Notes |
|--------|------|--------|-------|
| `model_monitor_trust_score` | Gauge | `window` | Per rolling window |
| `model_monitor_drift_score` | Gauge | `window` | Mean PSI |
| `model_monitor_f1` | Gauge | `window` | Weighted-average |
| `model_monitor_accuracy` | Gauge | `window` | |
| `model_monitor_n_batches` | Gauge | `window` | |
| `model_monitor_calibration_error` | Gauge | `window` | ECE |
| `model_monitor_decisions_total` | Counter | `action` | Monotonic; delta-incremented per scrape |
| `model_monitor_model_version_info` | Gauge | `version` | Label carrier; always 1.0; old label cleared on promotion |
| `model_monitor_f1_baseline` | Gauge | - | F1 at last promotion; `NaN` when no model active |
| `model_monitor_decision_latency_ms` | Histogram | `window` | 11 buckets: 10ms–2000ms |

**Counter correctness**: `_last_decision_counts` tracks the previous SQL
total per action type so `decisions_total` is incremented by the delta on
each scrape - never double-counted and always monotonically increasing,
regardless of how many records the database holds.

**Stale label cleanup**: when the active model version changes, the previous
`version` label set is removed from `model_version_info` via
`_model_version.remove(old_version)`. This prevents the Grafana version
annotation track from accumulating stale 1.0 values after promotions.

**`f1_baseline` NaN semantics**: the gauge is explicitly set to `float("nan")`
when no model is active. Prometheus text format emits `NaN` for these values;
Grafana treats them as "no data" so the baseline reference line disappears
rather than showing a stale value from the last promoted model in the same
process lifetime.

Add `metrics_path: /metrics` to any Prometheus scrape config to integrate
with Grafana, Datadog agent, or Alertmanager.

### Ingest API

`POST /metrics/ingest` accepts batch results from external inference
pipelines. Authenticated via `X-API-Key` header checked against the
`MONITOR_API_KEY` environment variable. Returns 503 if the variable is not
set (endpoint administratively disabled). Returns 401 for wrong key. Returns
422 for malformed payload (Pydantic validation). The ingest path records
`feature_drift_scores: None` since per-feature PSI requires the reference
distribution held server-side - only the aggregate drift score from the
caller is stored.

### Per-feature drift

`DriftMonitor.update()` sets `last_feature_scores: list[float]` after every
call once the sliding window is full. `Predictor.predict_batch()` reads
these and includes them in `last_metric_record.feature_drift_scores`.
`MetricsStore` serialises them to a JSON text column.

The Streamlit dashboard renders the most recent 50 batches × N features as
a colour-coded heatmap: green (PSI < 0.1), amber (0.1–0.2), red (> 0.2).
This makes it immediately visible which feature is drifting rather than only
seeing the mean PSI.

### Configuration

All policy thresholds are loaded from YAML files at startup via
`config/settings.py:load_config()`:

| File            | Controls                                                                    |
|-----------------|-----------------------------------------------------------------------------|
| `drift.yaml`    | `psi_threshold`, drift window size                                          |
| `retrain.yaml`  | `min_f1_gain`, `cooldown_batches`, `min_samples`, `min_stable_batches`      |
| `rollback.yaml` | `max_f1_drop`                                                               |
| `model.yaml`    | Model metadata (name, version, framework)                                   |

`rollback.yaml` was added after the initial release; `load_config()` falls
back to the hardcoded default (`max_f1_drop=0.15`) when the file is absent
so that older deployments continue to work without a migration step.

`monitoring/thresholds.py` contains two categories of constants:

- **Alerting constants** (`MIN_TRUST_SCORE`, `CRITICAL_TRUST_SCORE`): actively
  imported by `alerting.py`. Change these to tune alert sensitivity.
- **Reference constants** (everything else): documentation-only reflections
  of the YAML defaults. Not imported by production code.

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
- Falls back to full dataset when `n < 50` rows.
- `min_f1_improvement` uses `_IMPROVEMENT_EPS = 1e-9` to handle
  IEEE 754 rounding: `0.82 - 0.80 = 0.019999...` without the epsilon.

---

## Storage schema

Schema evolution is managed by `storage/migrations.py`. On every startup,
`MetricsStore.__init__` calls `run_migrations(engine)` which applies any
unapplied migrations in order. The current schema version is stored in the
`schema_version` table. Adding a column is a one-function addition to the
migration list; the `CURRENT_SCHEMA_VERSION == len(_MIGRATIONS)` invariant
is enforced by a test.

### `metrics_records`

| Column                       | Type       | Migration | Notes                  |
|------------------------------|------------|-----------|------------------------|
| `id`                         | INTEGER PK | baseline  |                        |
| `timestamp`                  | FLOAT      | baseline  | Unix epoch; indexed    |
| `batch_id`                   | TEXT       | baseline  | caller-assigned        |
| `n_samples`                  | INTEGER    | baseline  |                        |
| `accuracy`                   | FLOAT      | baseline  | [0, 1]                 |
| `f1`                         | FLOAT      | baseline  | [0, 1]                 |
| `avg_confidence`             | FLOAT      | baseline  | [0, 1]                 |
| `drift_score`                | FLOAT      | baseline  | mean PSI               |
| `decision_latency_ms`        | FLOAT      | baseline  |                        |
| `action`                     | TEXT       | baseline  | DecisionType           |
| `reason`                     | TEXT       | baseline  |                        |
| `previous_model`             | TEXT       | baseline  | nullable               |
| `new_model`                  | TEXT       | baseline  | nullable               |
| `behavioral_violation_rate`  | FLOAT      | v2        | nullable; BM branch           |
| `shap_attribution`           | TEXT       | v3        | nullable; JSON object         |
| `calibration_error`          | FLOAT      | v4        | nullable; ECE                 |
| `feature_drift_scores`       | TEXT       | v5        | nullable; JSON array          |
| `p95_latency_ms`             | FLOAT      | v6        | nullable; <20-sample batches  |
| `p99_latency_ms`             | FLOAT      | v6        | nullable; <20-sample batches  |
| `output_drift_score`         | FLOAT      | v7        | nullable; mean output PSI     |
| `output_drift_class_scores`  | TEXT       | v7        | nullable; JSON array          |
| `data_quality_score`         | FLOAT      | v7        | nullable; [0,1] aggregate     |
| `conformal_coverage`         | FLOAT      | v7        | nullable; labeled batches     |
| `conformal_set_size`         | FLOAT      | v7        | nullable; unlabeled proxy     |

### `decision_history`

| Column          | Type       | Notes                            |
|-----------------|------------|----------------------------------|
| `id`            | INTEGER PK |                                  |
| `timestamp`     | FLOAT      | indexed                          |
| `batch_index`   | INTEGER    | nullable                         |
| `action`        | TEXT       |                                  |
| `reason`        | TEXT       |                                  |
| `trust_score`   | FLOAT      | nullable                         |
| `f1`            | FLOAT      | nullable                         |
| `drift_score`   | FLOAT      | nullable                         |
| `model_version` | TEXT       | nullable                         |
| `metadata_json` | TEXT       | full `Decision.metadata` as JSON |

### `metrics_summary`

One row per aggregation window (upserted each pass).

### `metrics_summary_history`

Append-only; one row per window per aggregation run. Used for trend charts.

---

## Crash safety

**Retrain deduplication** (`SnapshotStore` - `behavior-monitoring` branch only):

> `SnapshotStore` and its write-ahead log pattern live in the
> `behavior-monitoring` branch.  The `main` branch records retrain
> decisions in `decision_history` instead; full crash-safe deduplication
> is a planned migration tracked in CONTRIBUTING.md.

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

- **Thresholds are hand-tuned.** Trust score weights and alert thresholds are
  reasoned defaults, not calibrated from historical data. The
  `notebooks/monitor_evaluation.ipynb` characterises the operating point
  empirically. A data-driven calibration step (e.g. optimising F-beta on a
  labelled evaluation window) is documented as future work.

- **Classification `Monitor` SDK and regression `RegressionMonitor` are
  independent entry points.** The core `Monitor` class (`monitor.py`) wraps
  classification models. `RegressionMonitor` (`monitoring/regression.py`) wraps
  regression models with Wasserstein-1 output drift, split-conformal prediction
  intervals, and MAE/RMSE trust components. They share the underlying monitoring
  primitives (PSI, MMD, data quality) but are separate classes rather than a
  unified generic interface. A fully generic `Monitor[T]` parameterised on output
  type would be cleaner but requires significant type machinery; the current two-
  class design is explicit and testable.

- **`MetricRecord` TypedDict has 24 fields (12 nullable).** At this size a
  dataclass with `field(default=None)` would be cleaner - typed fields with
  explicit defaults are easier to reason about than a flat TypedDict with many
  optional keys. The TypedDict was chosen for compatibility with existing
  SQLAlchemy ORM mapping code; a dataclass refactor would require an ORM
  mapping layer. Acknowledged as technical debt.

## New modules (v8)

### `monitor.py` - public SDK

Exposes `Monitor`, `MonitorConfig`, and `BatchResult` as the package-level public
API (`from model_monitor import Monitor`). Accepts any sklearn-compatible model or
callable, wires all monitoring components automatically, and returns a `BatchResult`
with `trust_score`, `drift_score`, `is_joint_drifting`, and `causal_summary` per
batch.

Design decision: the SDK is a *wrapper*, not a replacement for the internal
`Predictor`. `Predictor` is an internal class tuned for the simulation loop and
full API stack. `Monitor` is a clean external interface with minimal required
config, sensible defaults, and no dependency on `AppConfig` internals beyond the
minimal fields needed.

### `monitoring/mmd.py` - kernel MMD two-sample test

Implements the unbiased MMD² estimator from Gretton et al. (2012) with a
permutation-test p-value. Key implementation choices:

- **Unbiased estimator**: zero diagonal in K_XX and K_YY terms. The biased
  estimator has lower variance but is biased under H₀, which would inflate
  false-positive rates in permutation tests.
- **Median bandwidth heuristic**: σ² = median(‖xᵢ−xⱼ‖²)/2 over pooled sample.
  Estimated once at construction time. Per-batch re-estimation would produce
  inconsistent p-values across batches as the production distribution drifts.
- **Phipson & Smyth p-value correction**: p = (#{perm ≥ obs} + 1) / (B + 1).
  Avoids p=0 which would claim infinite evidence against H₀ from finite data.
- **Subsampling guard**: capped at `max_samples=1000` before kernel matrix
  computation. Keeps the O(n²) cost bounded; typical batch sizes are well within
  this limit.

The test `test_detects_joint_drift_invisible_to_marginals` is the key property test:
correlation flip ρ=+0.9 → ρ=−0.9 is undetectable by PSI (both marginals remain
N(0,1)) and reliably detected by MMD. This is the primary reason for the module's
existence.

### `monitoring/regression.py` - regression monitoring

Three components:

**`wasserstein1_distance`**: closed-form W₁ via quantile interpolation. No
parameters. Quantile-interpolates to a common grid of `min(n_ref, n_prod, 500)`
points so unequal-length arrays are handled without resampling.

**`RegressionConformalMonitor`**: split-conformal intervals with the finite-sample
correction level = `(1 + 1/n_cal) × (1 − alpha)`. The correction ensures
empirical coverage is exactly ≥ 1−alpha rather than ≥ 1−alpha − 1/n_cal for
finite calibration sets. The `monitor()` method uses a Wilson binomial confidence
interval to flag genuine under-coverage while avoiding false positives from
sampling noise on small batches.

**`compute_regression_trust_score`**: replaces accuracy/F1 components with
MAE/RMSE components that degrade linearly as error approaches the training-time
baseline. Custom weight dicts are validated but not required to sum to 1 (partial
weight specifications are supported, remainder defaults to zero penalty).

### `monitoring/cusum.py` - CUSUM sequential change-point detection (v11/v12)

Page–Hinkley CUSUM for detecting the exact batch where a sustained shift begins.

**Why CUSUM alongside MMD and PSI?** Batch tests (PSI, MMD) ask "did the
distribution change over this window?" - they need the full window to be drifted
before firing. CUSUM accumulates evidence across batches and fires at the first
batch where sustained evidence exceeds a threshold. For a drift that started two
batches ago, CUSUM fires in batch 3 while PSI on window=5 fires in batch 5.

**Reference mean auto-estimation.** A critical design decision: `reference_mean`
is *not* hardcoded. Hardcoding `reference_mean=0.0` causes every stable batch to
accumulate in `s_pos` (since real PSI is 0.01–0.08, never zero), producing false
alarms within 25 batches. Instead, `CUSUMDetector` construction is deferred until
`max(3, cusum_warmup)` PSI observations are collected from the actual deployment.
`reference_mean = mean(observed PSI)` is then set and the detector starts running.
This makes CUSUM self-calibrating: no user configuration required beyond `delta`
and `threshold`.

**`Monitor.on_alarm()`.** Registers Python callbacks that fire when any alarm
property (`is_drifting`, `is_joint_drifting`, `is_cusum_alarm`, `is_critical`) is
True. Callbacks run synchronously after `BatchResult` is built, before returning
to the caller. Exceptions in callbacks are swallowed - they never block inference.
Multiple callbacks are supported, firing in registration order.

**`Monitor.reset_after_retrain()`.** Clears the drift window buffer, CUSUM state
(detector and warmup buffer), and MMD batch counter so the post-retrain period is
measured cleanly. History is cumulative and not cleared. Without this, CUSUM's
accumulated pre-retrain evidence carries over and triggers a false alarm on the
first post-retrain batch.

---

## Bugs found and fixed by the test suite

**Floating point promotion threshold** (`training/promotion.py`):
`0.82 - 0.80` evaluates to `0.019999...` in IEEE 754 - less than `0.02`.
Fixed with `_IMPROVEMENT_EPS = 1e-9`.

**Entropy non-negativity** (`utils/stats.py`): additive EPS smoothing
caused a tiny negative result (`-1e-9`) for pure distributions. Fixed with
`max(0.0, ...)`.

**ORM state leaked into API responses** (`api/dashboard.py`): `__dict__`
on SQLAlchemy ORM rows includes `_sa_instance_state`. Fixed with
`_orm_to_dict()`.

**In-sample evaluation** (`training/retrain_pipeline.py`): candidate F1
was measured on training data. Fixed with a 20% held-out validation split.

**Decision metadata silently dropped** (`storage/decision_store.py`):
`Decision.metadata` was not persisted. Fixed by adding `metadata_json TEXT`.

**Cold-start trust score race** (`core/decision_runner.py`): first
aggregation pass could trigger a spurious retrain on an empty database.
Fixed with `if summary.n_batches == 0: continue`.

**`simulate_decision` used wrong trust score** (`api/dashboard.py`):
`/dashboard/decisions/simulate` approximated trust score with `avg_confidence`,
ignoring drift and latency. Fixed by calling `compute_trust_score()` with
the full set of fields, matching the live aggregation loop.

**Simulation loop never ran the decision engine** (`scripts/simulation_loop.py`):
`simulate_stream` constructed `Predictor(config=config)` without `f1_baseline`.
`predict_batch` requires `f1_baseline is not None` to engage the decision
engine - without it every batch returned `action="none"` with no drift or
trust scores computed. `make sim` printed `drift=0.000 trust=1.000` on every
line. Fixed by loading `baseline_f1` from `active.json` and passing it to
`Predictor`. Added `last_drift_score` and `last_trust_score` as first-class
`Predictor` attributes set unconditionally on every call.

**Prometheus `f1_baseline` gauge retained stale value** (`api/metrics.py`):
after a model promotion, the `f1_baseline` gauge held the promoted model's F1
for the lifetime of the process. If a subsequent scrape fired while no model
was active (e.g. between rollback and re-promotion), Grafana showed a stale
baseline reference line rather than "no data". Fixed by explicitly setting
`float("nan")` when `baseline_f1 is None`, which Prometheus text format emits
as `NaN` and Grafana treats as a gap in the series.

**`Inspector.from_engine(conn)` deprecated** (`storage/migrations.py`):
the initial migrations implementation called `Inspector.from_engine(conn)` -
an API removed in SQLAlchemy 2.0. On SQLAlchemy ≥2.0 this raised a
`TypeError` at startup for any database that had not yet received migration v2
(which checks for column existence). Fixed with `inspect(conn)`.

**Unpinned runtime dependencies** (`pyproject.toml`): all nine runtime
dependencies were listed without version bounds (`numpy`, `pandas`, etc.).
A `pip install` six months after a release could silently resolve to an
incompatible major version and produce runtime errors that never appeared in
CI. Fixed with `>=X.Y,<Z` ranges on all deps; `requirements.txt` updated to
match.

**Dashboard unauthenticated by default** (`api/dashboard.py`): the entire
`/dashboard/*` surface - including `POST /dashboard/decisions/simulate`,
`POST /dashboard/models/promote`, and `POST /dashboard/models/rollback` -
was unauthenticated with no mechanism to restrict access. Fixed with the
`MONITOR_DASHBOARD_KEY` env var and a single router-level `Depends()`.

**`simulate_decision` discarded `TrustScoreComponents`** (`api/dashboard.py`):
`POST /dashboard/decisions/simulate` called `compute_trust_score(...)` and
assigned the second return value to `_`.  The Streamlit simulation panel and
any API client calling this endpoint received only the scalar `trust_score` -
not the per-signal breakdown (accuracy, f1, confidence, drift, latency) that
explains *why* the engine would take an action.  On the BM branch,
`behavioral_violation_rate` was also absent from the simulation: it was never
passed to `compute_trust_score`, so the simulated trust score was always
computed without the behavioral component even when the latest record carried
a non-zero violation rate.  Fixed by capturing `trust_components`, passing
`behavioral_violation_rate` from the latest record, and including both in the
response.  Tests in `tests/test_simulate_endpoint.py` (main) and
`tests/test_simulate_decision_endpoint.py` (BM).

**`RawDataBuffer.add_batch` crashed silently on feature schema change**
(`monitoring/raw_data_buffer.py`, `inference/predict.py`): when the active
model was replaced with one trained on a different feature set, the next
`predict_batch` call raised `ValueError("Feature schema mismatch")`.  The
exception was caught by the broad `except Exception: pass` guard in
`predict_batch` - so inference was never blocked - but the buffer silently
accumulated no new data, meaning the next retrain would fall back to the
synthetic dataset instead of using real observed samples.  Fixed in two parts:
(1) `RawDataBuffer.reset_schema(new_feature_names)` discards stale rows,
resets the schema, and logs a WARNING with the row count being dropped;
(2) `Predictor.reload()` calls `reset_schema` when it detects that the new
model's `feature_names_in_` differs from the previous schema.

**`mypy ignore_missing_imports = true` was globally applied** (`pyproject.toml`):
the broad global flag silently suppressed type errors from any newly added
import that lacked stubs.  Three real errors were hidden: `stream_simulator.py`
returned `Any` from a pandas multiply, `streamlit_app.py` passed an incorrect
`Callable` type to `Styler.map`, and `test_predictor_failure_paths.py` passed
a wrong type without a suppression comment.  Fixed by replacing the global flag
with explicit `[[tool.mypy.overrides]]` blocks listing only the four packages
that genuinely lack stubs: `joblib`, `shap`, `sklearn`, `streamlit` (BM also:
`anthropic`).

**`ThresholdAdvisor` fed drifted batches despite comment saying "stable only"**
(`inference/predict.py`): the comment read *"Only records when no drift/reject/retrain"*
but no guard existed.  `observe()` was called unconditionally on every batch.
After drift injection, PSI values of 1.5–3.0 were fed to the advisor, producing
threshold recommendations of 0.56–3.81 - meaningless for any real deployment.
Fixed by adding `drift_score < cfg.drift.psi_threshold` as an explicit guard
before the `observe()` call.

**Causal attribution displayed as UNKNOWN** (`scripts/simulation_loop.py`): the
post-simulation causal display read from `predictor.last_metric_record`, which
stores `causal_drift_report` as a raw Python dict.  The parse logic checked
`isinstance(_causal_json, str)` and fell back to `_report = {}` (empty dict) for
the non-string case.  This caused `dominant_cause` to always be absent, showing
"UNKNOWN".  Fixed by handling both dict and JSON string.  Additionally the display
now scans the last 20 records from the metrics store to find the most recent batch
where causal attribution actually ran, rather than unconditionally reading the last
batch (which may be a reject with no attribution data).

**CUSUM `reference_mean=0.0` caused permanent false alarms** (`monitor.py`): see
CUSUM module documentation above.
