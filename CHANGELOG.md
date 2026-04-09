# Changelog

All notable changes to model_monitor are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Behavioral contracts for LLM output validation (see `behavior-monitoring` branch)

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
- 76 tests → 180 tests after cross-branch test porting and new tests
