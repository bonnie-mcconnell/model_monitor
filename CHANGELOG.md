# Changelog

All notable changes to model_monitor are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

No unreleased changes. See the [0.1.0] section for the current feature set.

---

## [0.1.0] - behavior-monitoring branch

### Added
- **`BehavioralContractRunner` wired into `predict_batch`** - `Predictor` now accepts
  an optional `behavioral_runner` and `behavioral_output` parameter. Evaluation runs
  within `behavioral_budget_ms` (default 50ms); exceptions are caught and logged so
  inference is never blocked by monitoring failures. Result available on
  `last_behavioral_record` for persistence via `BehavioralDecisionStore`.
- **`scripts/bench_evaluators.py`** - P50/P95/P99 latency benchmark for all evaluators;
  `--real-encoder` flag loads the production `all-MiniLM-L6-v2` model.
- **`CHANGELOG.md`** - Keep a Changelog format; every bug, fix, and design decision
  documented with rationale.
- **`[project.scripts]` entry points** - `model-monitor-train` and `model-monitor-sim`
  usable after `pip install`.
- **Mermaid architecture diagrams** - render natively on GitHub.
- **Behavioral contracts system** - versioned YAML contracts with four
  evaluators: `JsonValidityEvaluator`, `JsonSchemaEvaluator`,
  `ToneConsistencyEvaluator`, `LLMJudgeEvaluator`
- `ToneConsistencyEvaluator` - cosine similarity between output embedding
  and centroid of reference embeddings; uses injected `TextEncoder`
  Protocol so tests run without a model download
- `LLMJudgeEvaluator` - structured consistency verdict from an injected
  `LLMClient`; `MockLLMClient` for tests, `AnthropicLLMClient` for
  production
- `BehavioralContractRunner` - evaluates outputs against a contract,
  returns an immutable `DecisionRecord` with full evaluator provenance
- `BehavioralDecisionStore` - append-only SQLite persistence with
  idempotent writes; provides `violation_rate(since_ts)` for trust score
  integration
- `diff_decisions` - deterministic diff between two `DecisionRecord`s
  for regression detection between model versions
- `SnapshotStore` - write-ahead log for crash-safe retrain deduplication
- `POST /metrics/ingest` - authenticated API endpoint for external
  inference pipelines; `X-API-Key` auth, range-validated metric fields,
  503 when key is unset
- `AlertCooldownTracker` class - injectable cooldown state replacing the
  original module-level mutable dict; eliminates the test isolation
  problem and makes alert suppression testable without module introspection
- `metadata_json TEXT` column on `decision_history` - `Decision.metadata`
  is now serialised on every `record()` call, making the audit log
  complete and recoverable
- `MetricsSummaryHistoryStore.list_history()` - public read API with
  session expunge; eliminates the `_session_factory` private-attribute
  access that was present in the dashboard endpoint
- `_schedule_execution()` helper in `aggregation.py` - wraps
  `asyncio.create_task` with `add_done_callback` so executor failures are
  logged rather than silently discarded at GC time
- `scripts/bench_evaluators.py` - P50/P95/P99 latency benchmark for all
  evaluators; `--real-encoder` flag loads production model

### Changed
- `simulate_decision` endpoint now reads live metrics and active baseline
  F1 instead of hardcoded example values
- `inference/predict.py` - `predict_batch` now computes `accuracy` and
  calls `compute_trust_score` with the full six-component formula instead
  of a drift-only proxy; removes the `assert baseline is not None` that
  violated the no-bare-assert standard
- `DecisionAnalytics` - replaced `pandas` dependency with
  `collections.Counter` and a plain slice; removes a 50MB transitive
  dependency from the analytics layer
- `decision_store.tail()` - rows are expunged from the session before
  return; prevents potential `DetachedInstanceError` on deferred access
- `simulation_loop.py` - `__main__` block wrapped in `main()` function;
  `[project.scripts]` entry point now works correctly after `pip install`
- `pyproject.toml` - added `authors`, `license`, `classifiers`,
  `[project.urls]`, `[project.scripts]`; moved `matplotlib` to optional
  `[ui]` extra; added `mypy files = ["src", "tests"]`
- CI matrix expanded to Python 3.10 / 3.11 / 3.12; PR trigger now
  covers `behavior-monitoring` branch; `mypy` step covers `tests/`
- Coverage threshold enforced at 88%; `make coverage` target added

### Fixed
- **IEEE 754 float promotion** (`training/promotion.py`): `0.82 - 0.80`
  evaluates to `0.019999...` -  less than `0.02`. Candidates improving by
  exactly `min_improvement` were silently rejected. Fixed with
  `_IMPROVEMENT_EPS = 1e-9`.
- **Entropy non-negativity** (`utils/stats.py`): EPS smoothing produced
  `-1e-9` for pure distributions. Fixed with `max(0.0, ...)`.
- **ORM state leaked into API responses** (`api/dashboard.py`): `__dict__`
  on SQLAlchemy rows leaks `_sa_instance_state`. Fixed with
  `_orm_to_dict()`.
- **In-sample evaluation** (`training/retrain_pipeline.py`): candidate F1
  was measured on training data. Fixed with a 20% held-out validation
  split.
- **Decision metadata silently dropped** (`storage/decision_store.py`):
  `Decision.metadata` was not persisted. Fixed by adding `metadata_json`
  TEXT column.
- **Bare `assert` in source** (`inference/predict.py`,
  `core/decision_engine.py`): replaced with explicit `ValueError` and
  conditional narrowing respectively.
- **Inline import in dashboard** (`api/dashboard.py`): moved to
  module-level; private `_session_factory` access eliminated.
- **Hardcoded simulation endpoint**: `simulate_decision` now reads from
  live `MetricsStore` and `ModelStore`.
- **Module-level `ModelStore` instantiation** in `api/health.py` and
  `api/ingest.py`: replaced with lazy singletons to prevent import-time
  file system side effects.
- **Double module docstring** (`monitoring/thresholds.py`): merged two
  stacked docstrings into one.
- **`test_predictor_failure_paths.py` weak assertion**: replaced
  `assert decision.action in {"none", "retrain", "rollback"}` with four
  focused tests including a precise catastrophic-drop assertion.
- **`conftest.py` return type**: `create_test_tables` annotated
  `Generator[None, None, None]` instead of `-> None` on a generator.
- **`DecisionRunner` cold-start race**: `n_batches == 0` guard prevents
  `trust_score=0.0` default from triggering a spurious retrain on first
  startup.

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
- 76 tests → 320 tests after cross-branch test porting and new tests