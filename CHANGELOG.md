# Changelog

## [14.0.0] - predict_one, mypy tests/, .gitignore, benchmark table

### Added

- `Monitor.predict_one(x, y_true, flush_every)` - per-request inference path.
  Accumulates individual rows in an internal buffer and flushes the full
  monitoring pipeline automatically once `flush_every` rows are collected
  (default 64). Returns the model's raw prediction immediately, before the
  flush, so request latency is unaffected by monitoring overhead.
- `Monitor.flush()` - explicit drain of the `predict_one` buffer. Returns the
  `BatchResult` for the flushed mini-batch, or `None` if the buffer was empty.
  Call at process shutdown to ensure no in-flight rows go unmonitored.
- Pending buffer serialised by `Monitor.save()` and restored by `Monitor.load()`
  so in-flight rows survive checkpoint/restart cycles.
- `reset_after_retrain()` now clears the pending buffer - pre-retrain rows
  belong to the old distribution and must not contaminate the first
  post-retrain monitoring window.
- 13 new tests covering all `predict_one`/`flush` edge cases: auto-flush,
  manual flush, empty flush, label accumulation, save/load round-trip, reset,
  shape validation, pandas Series input, trust/drift score bounds.
- Performance benchmark table in README - wall-clock numbers at five batch
  sizes for `predict()` and `MMDDriftDetector.test()`, plus `predict_one`
  amortised cost.

### Fixed

- **`.gitignore`** - the repo had no `.gitignore` at all. SQLite databases,
  trained model pickles, reference stats, Jupyter checkpoints, and Docker
  volumes were all untracked. Fixed with a comprehensive `.gitignore`.
- **`.gitkeep` files** created for `data/metrics/`, `data/models/`,
  `data/reference/` - these directories must exist on a clean clone.
- **`db.py` clean-clone failure** - `_ensure_database_directory()` now parses
  the SQLite URL and calls `mkdir(parents=True, exist_ok=True)` before
  creating the engine. `make test` now works without any manual setup steps.
- **Six inline imports in `monitor.py`** - `logging`, `deque`,
  `sklearn.metrics`, `DecisionType`, `MetricRecord`, `MetricsStore`,
  `ModelEvaluation`/`build_model_card` were all buried inside method bodies.
  All moved to top-level. Module-level `log = logging.getLogger(__name__)`
  added.
- **51 mypy errors in `tests/`** - all `str` → `DecisionType` conversions
  across 9 test files; `_make_record` / `_make_metric` TypedDict missing
  fields; `_softmax` `Any` return; `MetricRecord | None` unguarded indexing;
  unused `# type: ignore` comments. `make typecheck` now runs
  `mypy src/model_monitor/ tests/` and reports zero errors.
- **Stale test counts** in README, CONTRIBUTING.md, and Makefile on both
  branches now reflect actual numbers (806 main, 995 behavior-monitoring).
- **ARCHITECTURE.md trust score weight table** corrected to match the actual
  YAML (23%/18%/14%/18%/17%/5%/5% instead of stale 25%/20%/15%/20%/20%).
- **Notebook NameError** in `uci_adult_drift_demo.ipynb` cell 15 - standalone
  `threshold_advisor` variable replaced with `monitor._advisor`.
- **AI meta-comment** "genuinely differentiated feature that demonstrates
  production engineering thinking" removed from `promotion.py` docstring.
- **`pandas-stubs`** added to `[dev]` dependencies - both branches now report
  `mypy: no issues found` instead of 17 import-untyped errors.
- **CI updated** - explicit `mkdir -p data/metrics data/models data/reference`
  step added; mypy scope extended to `tests/`; matrix now covers Python
  3.10/3.11/3.12; coverage threshold enforced at 80%.
- **"What I'd do differently"** section substantially expanded with honest,
  specific engineering tradeoffs on SQLite, threshold calibration, behavioral
  contract streaming limitations, MAB promotion, online MMD, and coverage
  strategy.
- **ARCHITECTURE.md `TrustScoreConfig` section** corrected: said "Five component
  weights" (there are seven) and cited stale defaults of 0.30/0.25/0.15/0.20/0.10;
  corrected to the actual code defaults 0.23/0.18/0.14/0.18/0.17/0.05/0.05.
- **CONTRIBUTING.md** "five weights" corrected to "seven weights".
- **README.md** bug count corrected from 37 to 18; "monitoring runs asynchronously"
  corrected to accurate description of deferred-flush semantics.
- **`anthropic` mypy override** moved from inline `# type: ignore` in
  `llm_judge.py` to `[[tool.mypy.overrides]]` in `pyproject.toml` (BM branch),
  consistent with how all other optional deps are handled.
- **`tmp_path` fixture annotation** in `test_predict_one.py` corrected from
  `pytest.TempPathFactory` to `pathlib.Path`.
- **Six `__init__.py` files** that contained only `from __future__ import
  annotations` now have proper module docstrings.

**`ThresholdAdvisor` fed drifted batches (critical).** The comment in
`predict.py` read *"Only records when no drift/reject/retrain"* but no guard
existed - `observe()` was called on every batch unconditionally.  After drift
injection, PSI values of 1.5–3.0 were fed to the advisor, producing threshold
recommendations of 0.56–3.81.  A "calibrated" threshold of 3.81 tells an operator
their warn threshold should be 38× the default - meaningless.  Fixed by adding
`drift_score < cfg.drift.psi_threshold` as an explicit guard before `observe()`.

**Causal attribution showed UNKNOWN in simulation output (critical).** The post-sim
display read `predictor.last_metric_record.get("causal_drift_report")` which is a
raw Python `dict` (in-memory path).  The parse logic only handled `str` (SQLite
path) and silently fell back to `_report = {}`.  Fixed by handling both `dict` and
`str`.  Also fixed: the display now scans the last 20 records from `MetricsStore`
to find the most recent batch with causal data, rather than unconditionally reading
the final batch (which may be a reject with no attribution).

### Added

**`Monitor.on_alarm()` - alert callbacks without a server.** Registers Python
callables that fire when any of `is_drifting`, `is_joint_drifting`, `is_cusum_alarm`,
or `is_critical` is True.  Exceptions in callbacks are swallowed - they never block
inference.  Example: `monitor.on_alarm(lambda r: requests.post(SLACK_URL, ...))`.

**`Monitor.reset_after_retrain()`.** Clears drift window buffer, CUSUM cumulative
sums, and MMD batch counter for clean post-retrain measurement.

**`GET /dashboard/drift/population`.** Per-feature PSI trend over the last N batches.
Returns `psi_by_feature`, `max_psi_per_feature`, `mean_psi_per_feature`.

**MMD joint-drift alert in `check_alerts()`.** Sixth alert condition: fires when
`mmd_is_drift=True` with a structured WARNING log explaining that joint drift may
be invisible to per-feature PSI.

**README completely rewritten** with all new SDK features: `on_alarm()`,
`warm_up()`, `reset_after_retrain()`, CUSUM configuration, standalone detectors,
population drift investigation endpoint, training provenance (`write_model_card()`).
The monitoring layers table updated to five layers including CUSUM.

**ARCHITECTURE.md** updated: CUSUM module documented with reference-mean
auto-estimation rationale; three new bugs documented.

### Tests

| Branch              | v12 | v13 | New tests                                           |
|---------------------|-----|-----|-----------------------------------------------------|
| main                | 791 | 793 | MMD alert (2)                                       |
| behavior-monitoring | 985 | 987 | same                                                |



### Fixed

**CUSUM `reference_mean=0.0` false-alarm bug (critical).** The `Monitor` SDK
constructed `CUSUMDetector(reference_mean=0.0)`.  Real stable-period PSI is
0.01–0.08, never zero.  With `delta=0.02` and stable PSI of `0.04`, every single
batch added `0.02` to `s_pos`, triggering an alarm within 25 stable batches - on a
perfectly healthy system.  Fixed by deferring `CUSUMDetector` construction until
`max(3, cusum_warmup)` PSI observations have been collected, then setting
`reference_mean = mean(observed PSI)`.  The detector is now calibrated to this
deployment's actual stable-period variance.  Verified with a test that asserts zero
false alarms over 20 in-distribution batches.

**Notebook cells 13 and 15** - cell 13 accessed `report.dominant_cause` (AttributeError;
`causal_summary` is a dict with key `"dominant_cause"`), cell 15 used `threshold_advisor`
(NameError; the variable is `monitor._advisor` post-SDK-rewrite).  Both fixed.

**MonitorConfig / MonitorSummary docstrings** - `cusum_delta`, `cusum_threshold`,
`cusum_warmup` were undocumented in `MonitorConfig`.  `cusum_alarm_rate` was
undocumented in `MonitorSummary`.  Both corrected.

### Added

- **`Monitor.on_alarm(callback, *, fire_on=...)`** - register Python callbacks that
  fire when any of the configured alarm properties (`is_drifting`, `is_joint_drifting`,
  `is_cusum_alarm`, `is_critical`) is True.  Multiple callbacks are supported.  A
  failing callback never blocks inference.  Enables custom alerting (Slack, PagerDuty,
  email, database write) without running the FastAPI stack.  4 new tests.

- **`Monitor.reset_after_retrain()`** - clears the drift window buffer, CUSUM
  cumulative sums (and warmup psi list), and the MMD batch counter so the post-retrain
  period is measured cleanly against the new model's reference distribution.  Without
  this, CUSUM's accumulated evidence from pre-retrain drift carries over, causing
  immediate false alarms on the first post-retrain batch.  History and batch count are
  preserved (cumulative across the deployment lifetime).  4 new tests.

- **CUSUM state fully persisted across restarts** - `save()`/`load()` now serialises
  the warmup PSI buffer, estimated `reference_mean`, `s_pos`, `s_neg`, and `n`.
  Change-point detection is continuous across process boundaries.

- **`GET /dashboard/drift/population`** - per-feature PSI trend over the last N
  batches.  Returns `psi_by_feature` (feature → list of PSI per batch),
  `max_psi_per_feature`, and `mean_psi_per_feature`.  This is the investigation
  endpoint: it answers "was this a sudden spike or gradual drift, and which feature
  moved first?"

### Tests

| Branch              | v11 | v12 | New tests                                              |
|---------------------|-----|-----|--------------------------------------------------------|
| main                | 782 | 791 | on_alarm (4), reset_after_retrain (4), CUSUM calibration (1) |
| behavior-monitoring | 976 | 985 | same                                                   |



### Added

- **`CUSUMDetector`** (`monitoring/cusum.py`) - Page–Hinkley CUSUM sequential
  change-point detection for ML monitoring.  Detects the exact batch where a
  sustained distribution shift begins, with better statistical power than batch
  tests (PSI, MMD) on small samples: a batch test needs the full window to be
  drifted, whereas CUSUM can detect a change that started two batches ago.
  Implementation: dual-sided cumulative sums with auto-reset after alarm,
  configurable allowance `delta`, decision threshold `h`, warmup period, and
  direction gate (`"up"` / `"down"` / `"both"`).  The expected ARL₀ for the
  false-alarm test is ~500 batches.  Integrated into `Monitor` via
  `MonitorConfig.cusum_delta` / `cusum_threshold` (disabled by default, enabled
  by setting both > 0).  `BatchResult.is_cusum_alarm` and `BatchResult.cusum_result`
  expose the per-batch state.  20 tests covering statistical calibration, direction
  gating, auto-reset, and false-alarm rate.

- **MMD joint-drift alert** (`monitoring/alerting.py`) - `check_alerts()` now has
  a sixth condition: when `mmd_is_drift=True`, a `WARNING` level log is emitted
  with the p-value, current PSI, and a note that joint drift may be invisible to
  per-feature tests.  This was the highest-value missing alert: MMD fires when
  correlation structure shifts while all marginals look stable - the exact case
  where PSI-only monitoring would stay silent.

- **`Monitor.warm_up(X)`** - pre-fills the drift window with reference-distribution
  data without recording a `BatchResult` or incrementing `n_batches`.  Prevents
  the artificially clean first `drift_window - 1` batches that occur when a fresh
  monitor is deployed mid-stream.  4 new tests.

- **`MonitorSummary` typed dataclass** - `Monitor.summary()` now returns a fully
  typed `MonitorSummary` dataclass instead of `dict[str, object]`.  All fields are
  IDE-completable and type-safe.  New fields: `mmd_drift_rate`, `latest_mmd_p_value`,
  `cusum_alarm_rate`.  Breaking change: callers using `summary()["n_batches"]` syntax
  should switch to `summary().n_batches`.

- **CUSUM state in `save()`/`load()`** - cumulative sums `s_pos` and `s_neg` are
  serialised so change-point detection is continuous across process restarts.

### Tests

| Branch              | v10 | v11 | New tests                                            |
|---------------------|-----|-----|------------------------------------------------------|
| main                | 751 | 782 | CUSUM unit (20), SDK CUSUM (4), warm_up (4), MonitorSummary (3), alerting (existing) |
| behavior-monitoring | 945 | 976 | same                                                 |



This release closes the gap between "features that exist" and "features that work
end-to-end." Every component added in v8–v9 is now wired through the complete
data pipeline: inference → SQLite → aggregation → Prometheus → Streamlit.

### What was broken and is now fixed

**MMD results not persisted (critical).** The `MMDDriftDetector` ran on every batch
and returned results in `BatchResult.mmd_result`, but those results were never
written to the database. `mmd_p_value` and `mmd_is_drift` were absent from
`MetricRecord`, the ORM, and `_aggregate_records`. The Prometheus gauges for MMD
were registered but would never fire. Fixed by adding the fields at every layer:
`MetricRecord` TypedDict → `MetricsORM` → `metrics_store.write()` → `_orm_to_dict()`
→ `_aggregate_records()` → `AggregatedSummary` → Prometheus exporter. MMD results
from `Monitor.predict()` and `Predictor.predict_batch()` now flow to the database
and appear in Grafana without any additional configuration.

**ModelCard never written (critical).** `training/model_card.py` was fully
implemented and the `/dashboard/models/{version}/card` API endpoint existed, but
`DefaultModelActionExecutor._handle_retrain()` never called `build_model_card()`.
Every request to the card endpoint returned 404. Fixed: `_write_model_card()` now
runs immediately after `store.promote_candidate()` returns the version number.
Best-effort: write failures are logged at WARNING and never block promotion.

**Regression API always returned `available: False` (critical).** The
`/dashboard/regression/latest` endpoint does `getattr(pred, "_regression_monitor", None)`,
but `Predictor` had no `_regression_monitor` attribute. Added `regression_monitor:
object | None = None` to `Predictor.__init__`. The regression endpoints are now
genuinely reachable when a `RegressionMonitor` is wired into a `Predictor`.

**Streamlit MMD panel fetched a nonexistent endpoint.** The panel called
`DASHBOARD + "/metrics/history?limit=50"` - that route does not exist. The correct
endpoint is `/metrics/tail`. Fixed.

**Drift window not saved in `Monitor.save()`.** `save()` checked
`hasattr(self._drift_monitor, "_window_arrays")` - that attribute does not exist.
`DriftMonitor` uses `self.buffer` (a `collections.deque`). The drift window was
silently not serialised. After a `load()`, PSI restarted from a cold empty window.
Fixed: `save()` serialises `list(self._drift_monitor.buffer)`; `load()` restores
it via `deque(restored_bufs, maxlen=window)`.

**Dead code in `Monitor.__init__`.** `AppConfig`, `RetrainConfig`, and `RollbackConfig`
were constructed solely to pass `AppConfig.drift` to `DriftMonitor`. `RetrainConfig`
and `RollbackConfig` were never read again. Removed; `DriftMonitor` now receives a
`DriftConfig` directly.

### Added

- **`Monitor.write_model_card()`** - public SDK method for recording training
  provenance when the internal retraining pipeline is not in use. Writes a
  `ModelCard` JSON capturing dataset hash, feature schema, evaluation F1, and
  arbitrary extra metadata. F1 is inferred from monitoring history when not
  supplied. Five new tests.

- **`Predictor.regression_monitor`** - optional `regression_monitor` parameter
  accepted by `Predictor.__init__`. Any object with `.history` and `.summary()`
  (i.e. `RegressionMonitor`) can be attached. The existing regression API
  endpoints now read from this attribute.

- **`mmd_p_value` / `mmd_is_drift` in `MetricRecord` and ORM.** Schema-level
  change: two new nullable columns in the `metrics` table. Existing databases
  are handled gracefully via SQLAlchemy's `nullable=True` with `create_all()`.

### Tests

| Branch              | v9  | v10 | New tests                                             |
|---------------------|-----|-----|-------------------------------------------------------|
| main                | 742 | 751 | ModelCard @ promotion (2), write_model_card (5), drift window save (2) |
| behavior-monitoring | 945 | 945 | same (BM already at 945)                              |



### Added

- **`Monitor.save()` / `Monitor.load()`** - persist and restore the full monitoring
  state (drift window buffers, threshold advisor observations, MMD bandwidth, batch
  history) to a JSON file.  Makes the SDK production-usable in containerised
  deployments where the process restarts between batches.  State is a self-contained
  JSON file; model weights and reference data are passed at load time.

- **`ModelCard`** (`training/model_card.py`) - training provenance attached to every
  promoted model version.  Records dataset hash (SHA-256 / float64, consistent with
  `RetrainEvidenceBuffer`), feature schema (name, dtype, min, max, null rate), held-out
  evaluation metrics, bootstrap CI bounds, promotion reason, pipeline version, and
  arbitrary extra metadata.  Serialisable to / from JSON.  Exposed via:
  - `GET /dashboard/models/{version}/card` - card for a specific version
  - `GET /dashboard/models/cards/all` - all available cards sorted by version

- **Prometheus gauges** - `model_monitor_mmd_p_value`, `model_monitor_mmd_is_drift`,
  `model_monitor_regression_mae`, `model_monitor_regression_rmse`,
  `model_monitor_regression_wasserstein`.  MMD and Wasserstein results are now visible
  in Grafana without any custom panel setup.

- **Streamlit sections 14 and 15** - MMD joint-drift panel (p-value trend, drift
  indicator) and Regression Monitor panel (W₁, MAE, RMSE, conformal interval coverage).

- **Regression API endpoints** - `GET /dashboard/regression/latest` and
  `/dashboard/regression/summary`, returning Wasserstein distance, MAE, RMSE, trust
  score, and conformal coverage rate for the most recent regression batch.

- **Notebook rewritten to use the SDK** (`uci_adult_drift_demo.ipynb`) - cells 8–11
  replaced: now calls `Monitor(clf, reference_data=X_train, ...)` on one line and
  `monitor.predict(X_batch)` per batch.  The manual 7-component wiring is replaced by
  a 12-line `MonitorConfig` block.  Lower-level components (causal Granger matrix,
  threshold advisor, per-feature PSI breakdown) are still demonstrated separately.

### Fixed

- **`is_healthy` hardcoded threshold** - was `return self.trust_score >= 0.70`,
  ignoring `MonitorConfig.trust_warn`.  Now uses the configured value.

- **`is_critical` added** - `BatchResult.is_critical` uses `MonitorConfig.trust_critical`
  (previously the field existed in config but was unused).

- **Double accuracy / F1 computation** - when `db_path` is set and `y_true` is supplied,
  accuracy and F1 were computed twice.  Second computation removed; results reused.

- **Full-dataset `predict_proba` at init** - `Monitor.__init__` was calling
  `predict_proba` on the entire `reference_data` array to initialise the output drift
  monitor.  For 100k-row training sets this took several seconds.  Now subsampled to
  ≤ 2000 rows, which is sufficient for PSI bin characterisation.

- **Silent `MetricsStore.write` failures** - bare `except: pass` replaced with a
  `logging.warning` so dropped records are visible in application logs.

- **`mmd_every` config** - `MMDDriftDetector` previously ran on every batch. Added
  `MonitorConfig.mmd_every` (default: 1) to run MMD every N batches, bounding latency
  in high-throughput settings.

- **Module docstring typo** - `monitor.py` docstring showed `preds, confs = monitor.predict()`
  which unpacks a `BatchResult` incorrectly.  Fixed to show the correct usage.

- **`psi_per_feature` on `BatchResult`** - per-feature PSI breakdown previously only
  accessible via `monitor.history[0]["feature_drift_scores"]`.  Now exposed directly
  as `result.psi_per_feature: dict[str, float] | None`.

### Tests

| Branch              | v8  | v9  | New tests                                            |
|---------------------|-----|-----|------------------------------------------------------|
| main                | 703 | 742 | SDK fixes (11), model card (28), save/load (6)       |
| behavior-monitoring | 899 | 938 | same additions                                       |



### Added

- **`Monitor` SDK** (`monitor.py`, `__init__.py`): public wrap-any-model interface.
  Accepts any sklearn-compatible model (or any `predict`/`predict_proba` callable),
  wraps the full monitoring stack, and exposes a clean five-line API::

      from model_monitor import Monitor
      monitor = Monitor(clf, reference_data=X_train, feature_names=cols)
      result  = monitor.predict(X_batch, y_true=y_batch)
      print(result.trust_score, result.is_joint_drifting)
      print(monitor.report())

  The SDK handles reference distribution fitting, component wiring, and optional
  SQLite persistence automatically. Feature names are inferred from DataFrame
  columns when not supplied.

- **`MMDDriftDetector`** (`monitoring/mmd.py`): kernel Maximum Mean Discrepancy
  (MMD) two-sample test for joint distribution shift. Detects cases where the
  joint distribution P(X₁, …, Xd) has changed even when all marginal distributions
  look flat - the dominant failure mode of univariate monitoring in production.
  Implementation: unbiased MMD² estimator with permutation-test p-values, median
  bandwidth heuristic (Gretton et al., 2012), and O(n²) subsampling guard.
  Integrated into `Monitor.predict()` as `result.is_joint_drifting`.

  The test suite includes `test_detects_joint_drift_invisible_to_marginals`, which
  verifies detection of a correlation sign flip (ρ=+0.9 → ρ=−0.9) that PSI
  cannot detect because both marginals remain N(0,1).

- **`RegressionMonitor`** (`monitoring/regression.py`): full monitoring stack for
  regression models. Components:
  - **Wasserstein-1 output drift**: closed-form earth mover's distance via sorted
    CDFs. No bandwidth selection, no binning artefacts. O(n log n).
  - **Conformal prediction intervals**: split-conformal method (Papadopoulos 2002)
    with the finite-sample correction q̂ = quantile(|y − ŷ|, (1+1/n)·(1−α)).
    Guarantees marginal coverage ≥ 1−α under exchangeability.
  - **Regression trust score**: MAE/RMSE components replace accuracy/F1; interval
    coverage rate replaces conformal set coverage.

- **Causal drift panel** (`ui/streamlit_app.py`, section 12): displays dominant
  cause, operator recommendation, and per-feature classification table. Reads from
  the existing `/dashboard/causal-drift/latest` endpoint.

- **Threshold advisor panel** (`ui/streamlit_app.py`, section 13): shows calibrated
  PSI and trust-score warn thresholds with a progress bar while collecting stable
  batches. Reads from `/dashboard/threshold-advisor/status`.

- **`ThresholdAdvisor` and `CausalDriftAttributor` wired into `simulation_loop.py`**:
  `make sim` now constructs both components, passes them to `Predictor`, and prints
  a post-simulation causal attribution table and threshold recommendations.

### Fixed

- **Notebook broken import** (`notebooks/uci_adult_drift_demo.ipynb`, cell 9):
  `compute_reference_stats` does not exist; replaced with the actual `DriftMonitor`
  constructor. The notebook now runs end-to-end without errors.

- **Notebook false claim about bundled data** (cell 2): the comment claimed
  `data/adult.csv` was bundled to avoid network dependency; it was not. Updated
  to correctly describe the download-and-cache pattern.

- **Dead code in `check_alerts`** (`monitoring/alerting.py`): `if summary is None:
  summary = {}` guard appeared after the `summary.get()` call that would already
  have raised `AttributeError` if `summary` were `None`. Guard removed.

- **`behavioral_runner` type annotation** (`inference/predict.py`, BM branch):
  annotated as `None` instead of `BehavioralContractRunner | None`. Fixed. Passing
  an actual runner instance previously failed type checking.

- **Makefile `notebook` target**: pointed at `drift_simulation.ipynb` (old notebook
  that computes PSI manually). Now points at `uci_adult_drift_demo.ipynb`.

### Tests

| Branch              | v7  | v8  | New tests                                      |
|---------------------|-----|-----|------------------------------------------------|
| main                | 600 | 703 | SDK (29), MMD (30), regression (37), endpoints (7) |
| behavior-monitoring | 796 | 899 | same additions                                 |



### Added

- **`CausalDriftAttributor`** (`monitoring/causal_drift.py`): Granger-causality
  attribution distinguishing genuine shift from pipeline failures. Classifies
  drifting features as ``genuine_shift``, ``pipeline_suspect``, or
  ``correlated_follower``. Reference: Granger (1969), Econometrica 37(3).

- **`ThresholdAdvisor`** (`monitoring/threshold_advisor.py`): adaptive threshold
  calibration from stable reference observations. Calibrated warn thresholds at
  the (1-alpha) percentile of stable-period PSI values. Eliminates alert fatigue.

- **`uci_adult_drift_demo.ipynb`** (`notebooks/`): full monitoring stack on the
  UCI Adult dataset with income drift simulating 30 years of economic change.
  Demonstrates that causal attribution correctly classifies the drift as genuine
  shift rather than a pipeline failure.

- **Rich webhook payload**: includes monitoring context (drift scores, conformal
  coverage, F1) and optional runbook URL via MODEL_MONITOR_RUNBOOK_URL env var.

- **Grafana layout fixed**: new panels repositioned to row 2 (immediately visible).

- **`ruff format` applied**: 116 files reformatted; format check enforced in CI.

### Test counts

| Branch              | v6  | v7  |
|---------------------|-----|-----|
| main                | 565 | 600 |
| behavior-monitoring | 761 | 796 |

---


## [6.0.0] - Functional completeness freeze

### Fixed (correctness bugs)

- **Per-sample timing regression** (`inference/predict.py`): the v5 per-sample
  timing loop called `model.predict_proba(X.iloc[[i]])` N times in a Python
  for-loop, making batch inference ~100x slower and producing misleading latency
  numbers. Reverted to vectorised batch inference; p95/p99 now computed via a
  separate 20-sample microbenchmark that measures per-request latency accurately.

- **`trust_score.yaml` stale keys**: the YAML still had `confidence` (5 keys)
  while `compute_trust_score()` had 7 components. `confidence` renamed to
  `calibration`; `data_quality` and `behavioral` added. `TrustScoreConfig`
  validator now checks all 7 weights sum to 1.0.

- **`output_drift_score` not reaching trust score**: `OutputDriftMonitor` wrote
  to `MetricRecord` but `compute_trust_score()` had no `output_drift_score`
  parameter. Fixed: combined drift = `max(input_psi, output_psi)`. A model
  with stable inputs but shifting output distribution now triggers the drift
  penalty.

- **`DecisionExplainer.rule_map` bare string literals**: `rule_map` used raw
  strings after `DecisionType` was converted to StrEnum. Replaced with
  `DecisionType.REJECT`, `DecisionType.ROLLBACK`, etc. Added
  `DecisionType.SYSTEM_ERROR` entry (previously returned `None`).

- **`retrain.yaml` missing `max_retrain_attempts`**: circuit breaker was
  implemented and tested but the operator-facing config file had no key.
  Added with full documentation of the guard and reset procedure.

### Added (observability completeness)

- **Prometheus gauges for new signals** (`api/metrics.py`):
  `model_monitor_output_drift_score`, `model_monitor_data_quality_score`,
  `model_monitor_conformal_coverage`, `model_monitor_conformal_set_size`.
  All four exported per aggregation window.

- **Four new Grafana dashboard panels** (`dashboards/grafana.json`): Output
  Distribution PSI, Data Quality Score, Conformal Coverage Rate, Conformal
  Prediction Set Size. With threshold bands matching alerting thresholds.
  Grafana version bumped; panel count 18 → 23.

- **New alerting rules** (`monitoring/alerting.py`, `monitoring/thresholds.py`):
  - `conformal_coverage < MIN_CONFORMAL_COVERAGE (0.85)` → warning
  - `data_quality_score < MIN_DATA_QUALITY_SCORE (0.80)` → warning
  - `output_drift_score > MAX_OUTPUT_DRIFT_SCORE (0.10)` → warning
  All use the existing cooldown tracker; all fire to log + webhook.

- **`/dashboard/config` exposes all 7 trust score weights and new thresholds**:
  `min_conformal_coverage`, `min_data_quality_score`, `max_output_drift_score`
  added to the alerting sub-dict. `confidence` key removed; `calibration` used.

- **`replay` CLI enriched with new monitoring context** (`cli/replay.py`):
  Table now shows InPSI, OutPSI, DQ (data quality), CvgR (conformal coverage)
  alongside trust score. JSON mode adds `input_drift_score`, `output_drift_score`,
  `data_quality_score`, `conformal_coverage` to each record. Operators can now
  correlate past drift events with monitoring signals without querying SQLite.

- **`simulation_loop.py` wires all three new monitors**: `OutputDriftMonitor`
  (reference probs from active model), `DataQualityMonitor` (feature bounds
  auto-computed as ±4σ of training distribution), `ConformalMonitor` (calibrated
  on first 20% of training population). `make sim` now shows OutPSI, DQ, CvgR
  columns in the live output table.

- **BM branch `BehavioralConfig`** (`config/settings.py`, `config/behavioral.yaml`):
  `budget_ms` (default 200ms, down from 500ms) and `ema_alpha` (default 0.30)
  are now configurable via YAML. `_DEFAULT_BEHAVIORAL_BUDGET_MS` constant removed.

- **Schema migrations 8 and 9**: add new signal columns to `metrics_summary`
  and `metrics_summary_history` tables. All idempotent.

- **`MetricsSummaryHistoryORM` extended**: four new nullable columns for the
  new monitoring signals. Replay CLI reads them via `getattr` with None fallback
  so it works on both old and new databases without error.

### Test counts

| Branch              | v5  | v6  | Delta |
|---------------------|-----|-----|-------|
| main                | 565 | 565 | 0     |
| behavior-monitoring | 761 | 761 | 0     |

No new tests were added in this release - all fixes closed bugs in existing
functionality. The test count is stable because every fix was covered by an
existing test that was previously failing or hidden by a wrong assertion.

---


All notable changes to model_monitor are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [5.0.0] - Monitoring stack expansion, StrEnum decisions, circuit breaker

### Added

- **`OutputDriftMonitor`** (`monitoring/output_drift.py`): PSI applied to the
  model's *output* probability distribution rather than input features. Detects
  prediction distribution shift (class imbalance, score compression) before
  batch F1 degrades. Reference bin edges are fixed at training time, mirroring
  the `DriftMonitor` design. Aggregate score is mean PSI across output classes;
  per-class breakdown stored on `last_class_scores`.

- **`DataQualityMonitor`** (`monitoring/data_quality.py`): leading-indicator
  monitoring for upstream data pipeline failures. Checks null rate, out-of-range
  values (configurable per-feature bounds), and schema consistency on each
  incoming batch. Returns `DataQualityReport` with `quality_score` in [0, 1].
  Score feeds `compute_trust_score()` as the new `data_quality` component.

- **`ConformalMonitor`** (`monitoring/conformal.py`): rigorous coverage
  monitoring using split conformal prediction (LAC variant). Calibrated on a
  held-out labeled set; monitors empirical coverage (`coverage_rate`) on labeled
  batches and prediction set size (`mean_set_size`) on unlabeled batches.
  Three-layer monitoring stack: input PSI → output PSI → conformal coverage.
  Reference: Angelopoulos & Bates (2021), arXiv:2107.07511.

- **Bootstrap CI in `compare_models()`** (`training/promotion.py`): when
  `n_bootstrap > 0` and predictions are provided, promotion is blocked unless
  the lower bound of the `(1-alpha)` bootstrap CI for the F1 improvement is
  positive. Prevents promoting a candidate that is only coincidentally better
  on a small holdout split. Paired bootstrap is used (same resampled indices
  for both models). `BootstrapCI` and `PromotionResult` dataclasses expose the
  full CI to callers.

- **p95 / p99 tail latency**: per-sample timing added to `predict_batch`
  (overhead ~1µs/sample). `p95_latency_ms` and `p99_latency_ms` stored in
  `MetricRecord` and `MetricsRecordORM`. The trust score `latency` component
  now uses p95 instead of mean; mean is the fallback when p95 is unavailable
  (batches < 20 samples).

- **Trust score `data_quality` component**: 5% weight; defaults to 1.0 (no
  penalty) when `DataQualityMonitor` is not configured.

- **Trust score `calibration` component** replaces `confidence`: ECE is a
  strictly better measure of confidence quality. Mapping: ECE=0 → 1.0,
  ECE=0.05 (Guo 2017 warning threshold) → 0.5, ECE≥0.10 → 0.0. `None`
  (no labels) → 0.8 (neutral discount).

- **`DecisionType` StrEnum** (`core/decisions.py`): `Literal["none", ...]`
  replaced with a `StrEnum` whose values ARE strings. Every existing string
  comparison, SQLite storage, JSON serialisation and Prometheus label continues
  to work unchanged. Adds tab-completion, membership testing, iteration, typo
  safety, and a single authoritative source for all valid action names.
  Python 3.10 compatibility shim included.

- **Retrain circuit breaker** (`core/decision_engine.py`): `max_retrain_attempts`
  in `retrain.yaml` (default 10) caps the total retrain decisions. When the cap
  is reached the engine emits `system_error` instead of `retrain`. Both the
  sustained-degradation path and the reject-escalation path share the same
  counter. Call `engine.reset_retrain_counter()` to re-open the breaker.
  Set `max_retrain_attempts: 0` to disable.

- **`scripts/demo_plot.py`**: standalone script generating the README hero
  image - a four-panel monitoring dashboard showing trust score, per-feature
  PSI, output drift + conformal set size, and tail latency across a 60-batch
  simulation with a drift-retrain-recovery arc. `make demo-plot` target added.

- **Schema migrations 6 and 7**: add `p95_latency_ms`, `p99_latency_ms`,
  `output_drift_score`, `output_drift_class_scores`, `data_quality_score`,
  `conformal_coverage`, `conformal_set_size` to the metrics table via
  idempotent `ALTER TABLE` migrations.

- **82 new tests** across 7 new test files: `test_output_drift.py`,
  `test_data_quality.py`, `test_conformal_monitor.py`,
  `test_bootstrap_promotion.py`, `test_new_columns_migration.py`,
  `test_trust_score_new_components.py`, `test_new_monitors_integration.py`,
  `test_circuit_breaker.py`.

### Changed

- `monitoring/alerting.py`: inline `import requests` moved to module-level
  `try/except ImportError` with `_REQUESTS_AVAILABLE` guard. Eliminates the
  "No inline imports" code standard violation.

- `api/schemas.py`: `DecisionType` duplicate `Literal` removed; the canonical
  `StrEnum` from `core/decisions.py` is imported directly. Pydantic validates
  incoming action fields against enum values.

- `api/dashboard.py`: `parse_decision_type()` now uses `DecisionType(value)`
  constructor instead of a hand-maintained set, so new enum members
  automatically extend the valid set.

- `ui/streamlit_app.py`: four new dashboard panels added (output drift, data
  quality, conformal coverage, tail latency). All inline `import requests`
  calls replaced with module-level guard.

- `pyproject.toml`: mypy `python_version` updated from `"3.10"` to `"3.11"` to
  enable full StrEnum type-checking (runtime shim preserves 3.10 compatibility).

### Test counts

| Branch              | Before | After |
|---------------------|--------|-------|
| main                | 456    | 565   |
| behavior-monitoring | 667    | 761   |

---

## [Unreleased] - export CLI, credibility polish, behavioral config

### Added

- **`model-monitor export`** (`cli/export.py`): write the full decision audit
  log to CSV or NDJSON.  Supports `--from`/`--to` date filtering, `--output`
  for file writing, `--format csv|json`.  Format inferred from file extension
  when unambiguous.  `make export` and `make export-json` targets added.
  `model-monitor-export` entry point wired in `pyproject.toml`.

- **`DecisionStore.query_range()`**: date-range filtered query over the
  decision audit log, both bounds optional and inclusive.  Used by the
  export CLI; 5 new tests in `tests/test_export.py`.

- **`conftest.py` store fixtures**: `metrics_store`, `decision_store`,
  `alert_store`, `model_store` pytest fixtures backed by isolated per-test
  SQLite files.  Eliminates the 62 repeated `Store(db_path=tmp_path/...)` 
  constructions across the test suite.

- **`docs/webhook_integration_examples.md`**: Slack, PagerDuty, and generic
  HTTP integration examples with exact payload schema, field table, and
  wiring instructions for `check_alerts`.

- **`docs/webhook_payload_example.json`**: machine-readable payload schema
  reference.

- **`docs/sim_output.png`**: terminal screenshot of an 80-batch simulation
  run.  Fixes the broken README reference that existed since initial publish.

- **`.env.example`**: documents `MONITOR_API_KEY`, `MONITOR_DASHBOARD_KEY`,
  and `GF_ADMIN_PASSWORD` with usage instructions and a key-generation
  command.

### Changed

- **`_BEHAVIORAL_EMA_ALPHA` moved to YAML config** (BM branch): was a
  module-level constant `0.3` with no config path - inconsistent with every
  other tunable parameter.  Added `behavioral_ema_alpha: float = 0.30` to
  `TrustScoreConfig` and `trust_score.yaml`.  EMA tests now read the value
  from `load_config()` so they stay in sync when the default is tuned.

- **`GET /dashboard/config`** now exposes `evidence_window` alongside the
  existing retrain parameters.  Response also includes a description of the
  dual-cooldown mechanism.

- **`simulate_decision` (BM)** now passes `behavioral_violation_rate` from
  the latest metric record into `compute_trust_score` and includes it in the
  `inputs` field of the response.

- **README Quick start** updated with CLI tools section and Docker/Grafana
  one-liner.  Test count updated to 456 / 667.

- **Grafana dashboard** `${DS_PROMETHEUS}` datasource variable added;
  all 18 panel target datasources updated from bare `"${datasource}"` string
  to `{"type": "prometheus", "uid": "${DS_PROMETHEUS}"}`.  Intro text panel
  added explaining trust score weights and alert thresholds.

- **`alembic` removed from dev deps** - was listed but never used; the
  lightweight `migrations.py` approach is intentional.  "Why not Alembic"
  rationale added to `migrations.py` docstring.

- **Replay CLI empty-range message** now includes timing guidance: which
  windows take 5 min vs 60 min to populate.

### Fixed

- **Inline `import logging` in `raw_data_buffer.py`**: `reset_schema()`
  used `import logging as _log` inside the method body - a style violation
  caught by the code standards ("no inline imports").  Moved to module level.

- **BM `simulate_decision`** missing `behavioral_violation_rate` in
  `compute_trust_score` call.  The BM endpoint was using the main-branch
  signature (no behavioral arg), so the simulated trust score never
  reflected the behavioral component.  Two tests that caught this were
  failing; both now pass.

- **CONTRIBUTING.md** still said API layer was excluded from coverage
  (accurate in v3, no longer true).  Removed the stale bullet and replaced
  with a correct statement listing the four test files that cover the API.


 [Unreleased] - observability stack, marker-based test filtering, coverage gate

### Added

- **Full observability stack in `docker-compose.yml`** (`--profile monitoring`):
  Prometheus (v2.51) and Grafana (v10.4) added as optional services.
  `docker compose --profile monitoring up --build` starts server + Prometheus +
  Grafana with auto-provisioned datasource and dashboard.  Grafana reads
  `dashboards/grafana.json` on startup; no UI configuration needed.
  `make monitoring` and `make monitoring-down` targets added.

- **Grafana provisioning configs** (`metrics/grafana/provisioning/`):
  `datasources/prometheus.yml` and `dashboards/model_monitor.yml` configure
  Prometheus as the default datasource and load the dashboard automatically.
  `metrics/prometheus.yml` scrapes `app:8000/metrics` every 15 seconds.

- **`@pytest.mark.slow` on `test_properties.py`** and `test_integration_e2e.py`:
  `make test-fast` now uses `-m "not slow"` instead of `--ignore=` paths, which
  is more composable and works correctly when new slow tests are added to any
  file.  9 slow tests deselected; 435 fast tests run in ~50 s on main.

- **`--tb=short` added to default `addopts`**: all pytest runs now show short
  tracebacks by default, matching what CI shows and reducing noise in local runs.

- **`notebooks/README.md`** updated with `monitor_evaluation.ipynb` entry,
  comparison table, and key results (0% FPR, 100% detection at 1σ).

### Changed

- **`docker-compose.yml`** rewritten: volumes mount `./data:/app/data` and
  `./models:/app/models` (not the previous `./data/metrics` only), so
  reference stats and logs persist across container restarts.
  `MONITOR_API_KEY` and `MONITOR_DASHBOARD_KEY` wired from host environment.

- **Coverage gate confirmed at 89%** on main (80% threshold). The previous
  `*/api/*` omission is removed; API layer is now covered by 29 tests across
  `test_api_routes.py`, `test_simulate_endpoint.py`, and `test_dashboard_auth.py`.

- **`filterwarnings`** in `pyproject.toml` extended with
  `ignore::DeprecationWarning:sqlalchemy` to suppress SQLAlchemy 2.x migration
  warnings that appear in every test run without adding signal.


 [Unreleased] - replay CLI, schema reset, mypy hardening

### Added

- **`model-monitor replay` CLI** (`cli/replay.py`): offline decision replay
  over stored `MetricsSummaryHistoryStore` records.  Supports `--from`,
  `--to` (ISO date/datetime), `--window` (5m / 1h / 24h), `--dry-run`,
  `--json` (newline-delimited for `jq` pipelines), and `--no-colour`.
  Recomputes trust_score from stored signals at replay time using the current
  `TrustScoreConfig`.  Wired into `pyproject.toml` as `model-monitor-replay`
  entry point; `make replay` and `make replay-dry` targets added to both
  Makefiles.  10 new tests in `tests/test_replay.py`.

- **`MetricsSummaryHistoryStore.query_range()`**: date-range filtered query
  returning rows oldest-first with inclusive bounds.  Used by the replay CLI.
  5 new tests covering bounds, ordering, and window filter.

- **`RawDataBuffer.reset_schema(new_feature_names)`**: gracefully discards
  stale rows and adopts a new schema when the model is retrained with a
  different feature set.  Logs a WARNING with the discarded row count.
  Wired into `Predictor.reload()` via `feature_names_in_` detection.
  5 new tests in `tests/test_raw_data_buffer.py`.

- **Updated Grafana dashboard** (`dashboards/grafana.json`): 17 panels (up
  from 9).  New panels: decision latency p50/p95/p99 timeseries, latency
  heatmap, behavioral violation rate, decision rate by action (per-minute),
  F1 baseline reference line (dashed).

### Changed

- **`mypy ignore_missing_imports = false`** on both branches: replaced the
  global `ignore_missing_imports = true` flag with explicit
  `[[tool.mypy.overrides]]` blocks for the four packages that lack stubs:
  `joblib`, `shap`, `sklearn`, `streamlit` (BM additionally: `anthropic`).
  Three real type errors surfaced and were fixed.

- **`MetricsSummaryHistoryStore._session_factory`** annotated as
  `Callable[[], Session]` so the factory is replaceable in tests without
  a `type: ignore` comment.

### Fixed

- `predict.py`: `except Exception: pass` for buffer schema-mismatch narrowed
  to `(ValueError, TypeError)`, and SHAP attribution narrowed to
  `(RuntimeError, ValueError, MemoryError)`.  Overly-broad swallows were
  hiding all errors including unexpected ones.

- `stream_simulator.py`: `DataFrame.__mul__` returns `Any` per pandas-stubs;
  fixed by constructing an explicit `pd.DataFrame(...)` to preserve the
  declared return type.

- `streamlit_app.py`: two `Styler.map` calls suppressed with targeted
  `type: ignore[arg-type]` (stubs overload is more restrictive than the
  actual API).


 [Unreleased] - production hardening & statistical evaluation

### Added

- **Schema migrations** (`storage/migrations.py`): append-only migration
  runner with a `schema_version` table. Replaces the `# NOTE: will move to
  migrations later` comment that survived 32 documented bug fixes. Idempotent
  on every startup; partial-migration state is safely resumed. Five migrations
  covering all historical column additions (`behavioral_violation_rate`,
  `shap_attribution`, `calibration_error`, `feature_drift_scores`). 8 new tests
  in `tests/test_migrations.py` including idempotency, partial-resume, column
  presence, and `CURRENT_SCHEMA_VERSION == len(_MIGRATIONS)` invariant.

- **Dashboard authentication** (`MONITOR_DASHBOARD_KEY` env var): all
  `/dashboard/*` routes now accept an optional `X-Api-Key` header enforced via
  a single router-level `Depends()`. When the env var is unset the dashboard is
  unauthenticated (local-dev default). When set, a missing or incorrect key
  returns HTTP 401 with a `WWW-Authenticate` header per RFC 7235. Key
  comparison is case-sensitive; the key is never echoed in error responses.
  Same pattern and rotation semantics as the existing `MONITOR_API_KEY` on the
  ingest endpoint. 13 new tests in `tests/test_dashboard_auth.py`.

- **`monitor_evaluation.ipynb`**: statistical characterisation of the PSI
  monitor. Answers the two questions that production deployment requires: (1)
  false-positive rate on stable data - **0%** across 200 independent trials;
  (2) detection latency vs shift magnitude - **100% detection at 1σ**, median
  **3 batches**, immediate (batch 0) at 2σ+. Includes PSI trajectory plots for
  three shift magnitudes, a detection-rate curve, a p5/p95 latency band, and an
  honest discussion of the sub-0.75σ blind spot and batch-size sensitivity.
  The analysis uses the same `compute_psi` implementation and reference bin
  edges as the production server - numbers are directly comparable to live
  deployment.

- **Real dataset support in `train.py`** (`--dataset breast-cancer`): loads
  the Wisconsin Breast Cancer dataset (569 samples, 30 real features, bundled
  with scikit-learn - no download). `make train DATASET=breast-cancer`
  demonstrates the monitor on a non-synthetic distribution. The dataset name is
  recorded in `data/reference/feature_schema.json` for the simulation loop.

- **Prometheus metrics rewritten with `prometheus-client`** (`api/metrics.py`):
  replaces the hand-rolled text-format emitter with the official library.
  Correct `Counter` (delta-increment across scrapes), `Gauge`, and `Histogram`
  (decision latency p50/p95/p99 with 11 configured buckets). `f1_baseline` is
  set to `NaN` when no model is active so Grafana shows no data point rather
  than a stale value. Label cleanup on model promotion. All Prometheus tests
  updated to decode `Response.body` correctly and assert `NaN` semantics.

- **`pre-commit` configuration** (`.pre-commit-config.yaml`): ruff lint and
  format, mypy, trailing-whitespace, YAML/TOML validation, debug-statement
  detection (`breakpoint()` / `pdb`), large-file guard (500 KB). Pinned to
  specific revs for reproducibility. Install with `pre-commit install` or run
  across all files with `make pre-commit`.

- **`make demo` target**: trains the initial model and prints four-terminal
  instructions for the full live stack (server + sim + dashboard). Runs
  correctly with `make train DATASET=breast-cancer` beforehand.

- **`make test-fast`, `make fmt`, `make pre-commit`** added to both branch
  Makefiles.

### Changed

- **Dependency version ranges pinned**: all nine runtime deps and twelve dev
  deps in `pyproject.toml` now carry `>=X.Y,<Z` bounds. Prevents silent
  breakage from incompatible major-version bumps. `requirements.txt` updated
  to match.

- **`prometheus-client>=0.17,<1`** added to runtime dependencies (replaces the
  ad-hoc Prometheus text formatting that was previously self-contained in
  `api/metrics.py`).

- **`pre-commit>=3.5,<4` and `alembic>=1.12,<2`** added to `[dev]` extras.

- **`make train`** now accepts `DATASET=` variable:
  `make train DATASET=breast-cancer`.

- Test counts updated everywhere: main **412**, behavior-monitoring **629**.

### Fixed

- `MONITOR_DASHBOARD_KEY` closes the unauthenticated dashboard surface
  documented as a known limitation in `SECURITY.md` since initial publication.

- `api/metrics.py`: `f1_baseline` gauge would retain its value from the
  previous model promotion across process-lifetime scrapes when no model was
  active (counter reset bug). Fixed with explicit `NaN` assignment.

- `storage/migrations.py`: used deprecated `Inspector.from_engine(conn)` API
  (removed in SQLAlchemy 2.x). Fixed with `inspect(conn)`.

- `storage/metrics_store.py`: the `# NOTE: schema creation is intentionally
  here for now. This will move to startup / migrations later.` comment is
  replaced by an actual call to `run_migrations(engine)`.


---

## [5.0.0] - Monitoring stack expansion, StrEnum decisions, circuit breaker

### Added

- **`OutputDriftMonitor`** (`monitoring/output_drift.py`): PSI applied to the
  model's *output* probability distribution rather than input features. Detects
  prediction distribution shift (class imbalance, score compression) before
  batch F1 degrades. Reference bin edges are fixed at training time, mirroring
  the `DriftMonitor` design. Aggregate score is mean PSI across output classes;
  per-class breakdown stored on `last_class_scores`.

- **`DataQualityMonitor`** (`monitoring/data_quality.py`): leading-indicator
  monitoring for upstream data pipeline failures. Checks null rate, out-of-range
  values (configurable per-feature bounds), and schema consistency on each
  incoming batch. Returns `DataQualityReport` with `quality_score` in [0, 1].
  Score feeds `compute_trust_score()` as the new `data_quality` component.

- **`ConformalMonitor`** (`monitoring/conformal.py`): rigorous coverage
  monitoring using split conformal prediction (LAC variant). Calibrated on a
  held-out labeled set; monitors empirical coverage (`coverage_rate`) on labeled
  batches and prediction set size (`mean_set_size`) on unlabeled batches.
  Three-layer monitoring stack: input PSI → output PSI → conformal coverage.
  Reference: Angelopoulos & Bates (2021), arXiv:2107.07511.

- **Bootstrap CI in `compare_models()`** (`training/promotion.py`): when
  `n_bootstrap > 0` and predictions are provided, promotion is blocked unless
  the lower bound of the `(1-alpha)` bootstrap CI for the F1 improvement is
  positive. Prevents promoting a candidate that is only coincidentally better
  on a small holdout split. Paired bootstrap is used (same resampled indices
  for both models). `BootstrapCI` and `PromotionResult` dataclasses expose the
  full CI to callers.

- **p95 / p99 tail latency**: per-sample timing added to `predict_batch`
  (overhead ~1µs/sample). `p95_latency_ms` and `p99_latency_ms` stored in
  `MetricRecord` and `MetricsRecordORM`. The trust score `latency` component
  now uses p95 instead of mean; mean is the fallback when p95 is unavailable
  (batches < 20 samples).

- **Trust score `data_quality` component**: 5% weight; defaults to 1.0 (no
  penalty) when `DataQualityMonitor` is not configured.

- **Trust score `calibration` component** replaces `confidence`: ECE is a
  strictly better measure of confidence quality. Mapping: ECE=0 → 1.0,
  ECE=0.05 (Guo 2017 warning threshold) → 0.5, ECE≥0.10 → 0.0. `None`
  (no labels) → 0.8 (neutral discount).

- **`DecisionType` StrEnum** (`core/decisions.py`): `Literal["none", ...]`
  replaced with a `StrEnum` whose values ARE strings. Every existing string
  comparison, SQLite storage, JSON serialisation and Prometheus label continues
  to work unchanged. Adds tab-completion, membership testing, iteration, typo
  safety, and a single authoritative source for all valid action names.
  Python 3.10 compatibility shim included.

- **Retrain circuit breaker** (`core/decision_engine.py`): `max_retrain_attempts`
  in `retrain.yaml` (default 10) caps the total retrain decisions. When the cap
  is reached the engine emits `system_error` instead of `retrain`. Both the
  sustained-degradation path and the reject-escalation path share the same
  counter. Call `engine.reset_retrain_counter()` to re-open the breaker.
  Set `max_retrain_attempts: 0` to disable.

- **`scripts/demo_plot.py`**: standalone script generating the README hero
  image - a four-panel monitoring dashboard showing trust score, per-feature
  PSI, output drift + conformal set size, and tail latency across a 60-batch
  simulation with a drift-retrain-recovery arc. `make demo-plot` target added.

- **Schema migrations 6 and 7**: add `p95_latency_ms`, `p99_latency_ms`,
  `output_drift_score`, `output_drift_class_scores`, `data_quality_score`,
  `conformal_coverage`, `conformal_set_size` to the metrics table via
  idempotent `ALTER TABLE` migrations.

- **82 new tests** across 7 new test files: `test_output_drift.py`,
  `test_data_quality.py`, `test_conformal_monitor.py`,
  `test_bootstrap_promotion.py`, `test_new_columns_migration.py`,
  `test_trust_score_new_components.py`, `test_new_monitors_integration.py`,
  `test_circuit_breaker.py`.

### Changed

- `monitoring/alerting.py`: inline `import requests` moved to module-level
  `try/except ImportError` with `_REQUESTS_AVAILABLE` guard. Eliminates the
  "No inline imports" code standard violation.

- `api/schemas.py`: `DecisionType` duplicate `Literal` removed; the canonical
  `StrEnum` from `core/decisions.py` is imported directly. Pydantic validates
  incoming action fields against enum values.

- `api/dashboard.py`: `parse_decision_type()` now uses `DecisionType(value)`
  constructor instead of a hand-maintained set, so new enum members
  automatically extend the valid set.

- `ui/streamlit_app.py`: four new dashboard panels added (output drift, data
  quality, conformal coverage, tail latency). All inline `import requests`
  calls replaced with module-level guard.

- `pyproject.toml`: mypy `python_version` updated from `"3.10"` to `"3.11"` to
  enable full StrEnum type-checking (runtime shim preserves 3.10 compatibility).

### Test counts

| Branch              | Before | After |
|---------------------|--------|-------|
| main                | 456    | 565   |
| behavior-monitoring | 667    | 761   |

---

## [Unreleased] - correctness & observability pass

### Added
- **`evidence_window` in `RetrainConfig`**: new YAML key separating "how many
  monitoring summaries before retrain triggers" from `min_samples` (raw training
  data threshold). Fixes the silent retrain skip - see bug 23 below.
- **Property-based tests** (`tests/test_properties.py`, 7 tests): Hypothesis
  covers PSI non-negativity, PSI(X,X)=0, trust score ∈ [0,1], trust monotone
  in drift and F1, decision engine always returns a known action, and engine
  rejects invalid trust_score. These verify mathematical invariants over
  thousands of random inputs, not just hand-chosen examples.
- **End-to-end integration tests** (`tests/test_integration_e2e.py`, 2 tests):
  boots a real model, runs the simulation with drift injection, asserts retrain
  fires, a new model is persisted to `ModelStore`, and "new model promoted"
  appears in output. The most important test - verifies the system self-heals.
- **`reload_reference()` on `Predictor`**: replaces the `DriftMonitor`'s
  reference distribution after promotion so future PSI scores measure drift
  against the new baseline, not the original training distribution.
- **`consecutive_rejects` field in `DecisionMetadata`**: audit trail for the
  sustained-reject escalation rule.
- **`slow` pytest mark registered** in `pyproject.toml` to suppress the
  unknown-mark warning on integration and property-based tests.

### Changed
- **`shap` moved to optional extras** (`pip install model-monitor[shap]`). Was
  in mandatory dependencies, pulling 100MB+ for users who never configure
  SHAP attribution.
- **`requests` lazy-imported** inside `safe_get`/`safe_post` in
  `streamlit_app.py`. `requests` is a `[ui]` extra - a top-level import caused
  `ImportError` on base installs.
- **`_configure_sim_logging` scoped to `model_monitor` namespace**: no longer
  calls `root.handlers.clear()`, which silently dropped pytest / uvicorn
  handlers. Logger now uses `logging.getLogger("model_monitor")` with
  `propagate=False`.
- **`hypothesis` added to `[dev]` extras**: enables property-based tests in CI.
- **`retrain.yaml` comment clarified**: `window` in `drift.yaml` is a sample
  count, not a batch count. The simulation uses `sim_drift_window` (batches).
- **CI coverage threshold synced**: `behavior-monitoring` branch CI now matches
  the Makefile at 88% (was 80%).
- **`ARCHITECTURE.md` SnapshotStore note**: clarifies `SnapshotStore` lives in
  the `behavior-monitoring` branch only.

### Fixed
- **Bug 23: Simulation printed "retrain" but never actually trained.**
  `RetrainEvidenceBuffer` was gated on `min_samples=1000`. With 80 batches
  the buffer held 80 entries - never ready. `evidence_window=3` now governs
  the evidence gate; `min_samples` governs training data quantity only.
- **Bug 24: `candidate_exists` not passed in aggregation loop.**
  `aggregate_once()` and `start_aggregation_loop()` never passed
  `candidate_exists` to the decision engine. The promote rule was dead code in
  the production path. Fixed: `candidate_exists=model_store.has_candidate()`.
- **Bug 25: `applymap` removed in pandas 3.x.**
  `streamlit_app.py` used `DataFrame.style.applymap` (removed in pandas 3.0).
  Raised `AttributeError` when the alert table had rows. Fixed: `.map()`.
- **Bug 26: Sustained rejection had no recovery path.**
  After severe drift, the system rejected every batch forever with no escape.
  Fixed: after `cooldown_batches` consecutive rejects the engine escalates to
  `retrain`, continuously attempting to adapt to the drifted distribution.
- **Bug 27 (sim): Promoted model never persisted to disk.**
  `simulation_loop.py` called `retrain_pipeline.run()` but never called
  `model_store.save_candidate()` or `model_store.promote_candidate()`. The
  console printed "↑ new model promoted" but nothing was written to disk;
  `predictor.reload()` loaded the old model. Fixed: explicit
  `save_candidate → promote_candidate` calls with version timestamp in output.

- **Bug 31: `simulate_decision` endpoint missing `candidate_exists`.**
  `POST /dashboard/decisions/simulate` always evaluated `candidate_exists=False`,
  so the promote rule was never reachable from the Streamlit simulation panel
  regardless of what was staged. Fixed: passes `model_store.has_candidate()`.
  Also surfaces `candidate_exists` in the response so operators can see why
  promote was or was not recommended.

- **Bug 32: `GET /config` returned hardcoded alerting thresholds.**
  The endpoint returned `0.70` and `0.60` literals instead of importing from
  `monitoring/thresholds.py`, where `MIN_TRUST_SCORE` and
  `CRITICAL_TRUST_SCORE` are defined. The two values could silently diverge
  if `thresholds.py` was updated. Fixed: imports and uses the constants.


### Fixed (production-readiness pass 3)

- **Bug 28: `startup.py` used `min_samples` for `RetrainEvidenceBuffer`.**
  The evidence buffer in the production server was gated on `min_samples=1000`
  (raw training data rows), not `evidence_window=3` (monitoring summaries).
  The live FastAPI server would never trigger a retrain regardless of how
  degraded the model became - the identical bug that was fixed in
  `simulation_loop.py` in pass 2. Fixed: `cfg.retrain.evidence_window`.

- **Bug 29: `assert_bounded("avg_drift_score", ..., hi=1.0)` crashed the
  aggregation loop when PSI exceeded 1.0.**
  PSI is unbounded - a value of 1.45 is normal under severe distribution shift
  (seen in the simulation from batch 44 onward). The invariant check would
  raise `InvariantViolation` at exactly the moment it most needed to keep
  running. Fixed: replaced with `assert_non_negative`, which is the only hard
  invariant on PSI.

- **Bug 30: `DecisionRunner.run_once()` missing `candidate_exists`.**
  The `promote` rule in `DecisionEngine` requires `candidate_exists=True` to
  fire. `aggregate_once()` was fixed in pass 2; `DecisionRunner.run_once()` -
  the synchronous alternative used by CLI tools and offline replay - was missed.
  Fixed: passes `model_store.has_candidate()` when a `model_store` is provided.

- **Bug 31: `assert_non_negative` only accepted `int`, crashed on `float`.**
  The `avg_drift_score` fix (bug 29) revealed that `assert_non_negative` had
  `value: int` in its signature. Fixed: broadened to `int | float`.

- **BM: `startup.py` did not wire `behavioral_store` or `snapshot_store`.**
  The `behavior-monitoring` branch server started without behavioral contract
  evaluation or crash-safe retrain deduplication. Both stores are now
  constructed and passed to `start_aggregation_loop`.



### Added (production-readiness pass 2)
- **Expected Calibration Error (ECE)**: added to `utils/stats.py` with full
  docstring and references. Flows through `predict_batch` →
  `last_metric_record.calibration_error` → `MetricsStore` (nullable Float
  column) → `AggregatedSummary.avg_calibration_error` → `MetricsSummaryORM`
  → Prometheus gauge `model_monitor_calibration_error` → Streamlit dashboard.
  Catches models with high F1 that are still badly calibrated.
- `MetricsStore.prune_before(cutoff_ts)`: deletes records older than a
  timestamp, returns count. Called automatically by the aggregation loop
  with `retention_days=30` default.
- `DecisionStore.count_by_action()`: SQL `GROUP BY` query returning per-action
  totals. Used by the Prometheus `decisions_total` counter so it is truly
  monotonically increasing at any scale.
- `GET /dashboard/config`: returns all live policy thresholds with descriptions
  (PSI threshold, min_f1_gain, max_f1_drop, cooldown, trust score weights,
  latency/drift→trust mappings, alerting thresholds).
- `GET /dashboard/health/detailed`: single-payload ops status - model version,
  trust scores per window, latest batch info, alert rates (1h/24h/critical 24h),
  total decision count.
- `GET /dashboard/models/compare?v1=X&v2=Y`: compares two archived versions
  using metadata stored at promotion time.
- 24 new tests in `test_production_readiness_2.py`.

### Fixed (production-readiness pass 2)
- `drift.yaml` window=500 made `DriftMonitor` return 0.0 for all 80 simulation
  batches - drift was never visible in `make sim`. Fixed with `sim_drift_window=5`
  default in `simulate_stream` (PSI fires after batch 5). Production deployments
  still use the configured 500-batch window via `sim_drift_window=0`.
- Prometheus `model_monitor_decisions_total` was not a real counter - it counted
  the last 10,000 rows in Python, meaning it could decrease after 10,000 total
  decisions. Fixed with `DecisionStore.count_by_action()` SQL `GROUP BY`.
- `docker-compose.yml` stale comment claiming POST /metrics/ingest was only on
  behavior-monitoring branch.
- `MetricsEventIn` had no field validation - callers could POST `accuracy=-5.0`
  or `f1=999.0` and corrupt trust scores. Fixed with Pydantic `ge`/`le` constraints.
- `DATABASE_URL` was hardcoded in `db.py`. Now reads from environment variable
  `DATABASE_URL` with fallback to the previous default.
- `_aggregate_records` used unweighted mean for accuracy and F1 - a 10-sample
  batch had equal weight to a 1000-sample batch. Fixed with sample-weighted mean
  (`n_samples`-weighted for accuracy/F1; unweighted remains correct for PSI/latency).
- Test counts: main 274, behavior-monitoring 408.

### Added
- `WebhookAlerter` in `monitoring/alerting.py`: outbound HTTP notifications
  to Slack, PagerDuty, or any JSON webhook endpoint. `severity_filter`,
  `timeout_s=2.0`, injectable `_post`, lazy `requests` import. 15 tests.
- `monitoring/windows.py`: single source of truth for aggregation window
  constants. `aggregation.py` imports from here; adding a window = 1 file.
- `config/rollback.yaml`: `max_f1_drop` configurable via YAML with backward-
  compatible fallback.
- `psi_bin_edges` stored in `reference_stats.json`: `compute_reference_stats()`
  writes training-time bin edges; `DriftMonitor` accepts `stored_bin_edges`
  and passes them to `compute_psi`, preserving the reference feature space.
- `make dashboard` target.
- **Per-feature PSI scores**: `DriftMonitor.last_feature_scores` set after
  every `update()` call. `MetricRecord.feature_drift_scores` field added.
  `MetricsRecordORM` has a `TEXT` column for the JSON-serialised list.
  `MetricsStore` persists and deserialises them on round-trip. Dashboard
  renders them as a colour-coded heatmap (green/amber/red per PSI threshold).
- **`Predictor.last_metric_record`**: full `MetricRecord` built after every
  `predict_batch()` call, populated unconditionally regardless of which
  decision branch fires. Includes `feature_drift_scores`.
- **Simulation loop → `MetricsStore`**: `simulate_stream` writes
  `predictor.last_metric_record` to `MetricsStore` after every batch.
  `make sim` now populates the dashboard in real time - the two were
  completely disconnected before this change.
- **`POST /metrics/ingest`** ported to main branch. Set `MONITOR_API_KEY`
  and POST batch results from any external inference pipeline.
- **Prometheus `GET /metrics` endpoint** (`api/metrics.py`): exports
  `model_monitor_trust_score`, `model_monitor_drift_score`, `model_monitor_f1`,
  `model_monitor_accuracy`, `model_monitor_n_batches` per window, plus
  `model_monitor_decisions_total` counters by action and
  `model_monitor_model_version_info`. No external dependency - plain text.
- **`AlertStore`** (`storage/alert_store.py` + ORM): persistent alert history.
  `check_alerts()` now accepts optional `alert_store` and calls
  `alert_store.record()` on every fired alert. Cooldown-suppressed alerts
  are not persisted. `AlertStore.count_since()` for rate queries.
- **`GET /dashboard/alerts/history`**: queryable alert history with
  severity/window/since_ts filters.
- **`GET /dashboard/models/active`** and **`GET /dashboard/models/versions`**:
  expose the active model metadata and full version archive.
- **`DecisionStore.count()`**: returns total decisions recorded; used by
  `simulate_decision` as `batch_index` so cooldown logic matches the live
  aggregation loop.
- **Streamlit dashboard rewritten** (9 sections): fixed broken auto-refresh
  (was calling `st.rerun()` unconditionally on every render cycle, burning
  CPU); per-feature drift heatmap; alert history with severity filter;
  model version timeline; Prometheus preview; quick-action sidebar with
  live simulate result.
- 65 new tests total across this and the previous session.

### Fixed
- `startup.py` hardcoded `min_samples=5`, bypassing `retrain.yaml` config
  (effective threshold was 5 records regardless of configured value).
  Fixed by calling `load_config()` at startup.
- `dashboard.py` created all stores at module import time - side-effecting
  SQLite files and `models/` directories on import. Fixed with lazy
  singletons (first-request construction), matching `health.py`.
- `simulate_decision` passed `batch_index=0` to the decision engine. With
  any prior retrain, `since_last = 0 - _last_retrain_batch < 0 < cooldown_batches`,
  so the ephemeral cooldown always fired and the endpoint always returned
  `action="none"` for retrain. Fixed using `DecisionStore.count()`.
- `dashboard.py` imported `RetrainPipeline` but never used it (dead import).
- Simulation loop (`make sim`) and dashboard (`make dashboard`) were
  disconnected - the sim wrote nothing the dashboard could read.
  Fixed by writing every batch to `MetricsStore`.

### Changed
- `aggregate_once()` and `start_aggregation_loop()`: added optional
  `alert_store: AlertStore | None` parameter.
- `check_alerts()`: added optional `alert_store` parameter.
- `compute_psi()`: added optional `bin_edges` parameter.
- `DriftMonitor.__init__`: added `stored_bin_edges` parameter; added
  `last_feature_scores: list[float]` attribute.
- `Predictor.__init__`: added `stored_bin_edges`, `last_metric_record`
  attributes alongside existing `last_drift_score` / `last_trust_score`.
- `MetricRecord` TypedDict: added `feature_drift_scores: list[float] | None`.
- `DecisionStore.__init__`: added optional `db_path` parameter for test
  isolation (same pattern as `MetricsStore`).
- `api/main.py`: registered `ingest` and `metrics` routers; added
  FastAPI title/description/version.
- Makefile test counts: main 250, behavior-monitoring 384.

---
  `compute_reference_stats()` now writes equal-frequency bin edges at
  training time. `DriftMonitor` accepts `stored_bin_edges: dict[int,
  np.ndarray]` and passes them to `compute_psi`, which uses them directly
  instead of recomputing from the reference array. This is the production
  path for the architecture guarantee that PSI bin edges are pinned to the
  training distribution.
- `make dashboard` target: runs the Streamlit monitoring UI.
- 39 new tests: PSI stored bin edges (5), `DriftMonitor` stored edges (2),
  rollback YAML loading (4), `compute_reference_stats` psi_bin_edges (7),
  `WebhookAlerter` (15), dual-cooldown disagreement (3),
  `Predictor` observability attributes (6) - `last_drift_score` /
  `last_trust_score` set unconditionally on every `predict_batch` call.

### Fixed
- `api/dashboard.py`: `/dashboard/decisions/simulate` approximated trust
  score with `avg_confidence`, ignoring drift and latency. Now calls
  `compute_trust_score()` with the full set of fields from the latest
  metric record, matching the live aggregation loop.
- `monitoring/thresholds.py`: module docstring now clearly distinguishes
  alerting constants (imported by `alerting.py`) from reference constants
  (documentation only, not imported). Eliminates confusion about which
  constants actually govern live policy.
- `scripts/simulation_loop.py`: `simulate_stream` never passed `f1_baseline`
  to `Predictor`, so `predict_batch` always returned `action="none"` with
  no real drift or trust scores computed. `make sim` printed
  `drift=0.000 trust=1.000` on every line - the simulation was decorative,
  not functional. Fixed by loading baseline F1 from `active.json` and
  passing it to `Predictor`. Also fixed metric reads: `drift_score` and
  `trust_score` are now read from `predictor.last_drift_score` /
  `predictor.last_trust_score` (new first-class attributes) rather than
  fragile `decision.metadata.get(...)` calls that only populated for
  certain action types.

### Changed
- `compute_psi()` signature: added optional `bin_edges: np.ndarray | None`
  parameter. When supplied, used directly; when absent, derived from
  `expected` (original behaviour preserved for backward compatibility).
- `DriftMonitor.__init__`: added optional `stored_bin_edges: dict[int,
  np.ndarray] | None` parameter, forwarded to `compute_psi` per feature.
- `Predictor.__init__`: added `stored_bin_edges` parameter (forwarded to
  `DriftMonitor`) and `last_drift_score` / `last_trust_score` attributes.
- `load_config()`: added optional `rollback_path` parameter; loads
  `rollback.yaml` when present.
- Makefile: `make test` count updated to reflect current suite size.

---

## [0.0.1] - main branch (classical monitoring baseline)

### Added
- PSI drift detection with reference bin edges stored at training time
- Trust score: bounded [0, 1] weighted combination of accuracy, F1,
  confidence, drift, and latency components
- Decision engine: pure policy function - reject on severe drift, rollback
  on catastrophic F1 drop, retrain on sustained degradation, promote on
  stability; dual cooldown (ephemeral + durable)
- `DecisionExecutor`: async orchestration with `asyncio.Lock`, SHA-256
  retrain idempotency key, `SnapshotStore` write-ahead log
- `MetricsStore`: cursor-based pagination; consistent under concurrent
  writes
- `ModelStore`: atomic rename promotion, crash-safe via `os.replace`,
  lazy singleton to avoid import-time side effects
- `RetrainPipeline`: 20% held-out validation split; both models evaluated
  on the same set for a fair comparison
- FastAPI app with `/health`, `/ready`, `/dashboard/*` endpoints
- Streamlit monitoring dashboard
- Simulation loop: 80-batch drift scenario with live terminal output
- 76 tests → 186 tests after cross-branch test porting and new tests