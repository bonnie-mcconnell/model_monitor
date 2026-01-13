# Model Monitor

Production-style ML model monitoring system with drift detection,
delayed labels, retraining, and promotion.

## Why this exists
<short paragraph about real ML systems drifting silently>

## Architecture
<diagram OR bullet list>

## Key Features
- Streaming inference simulation
- Drift detection (PSI)
- Metrics history tracking
- Automated retraining & promotion
- Health + readiness probes
- OpenAPI-powered API
- UI dashboard built on API

## How it works
1. Simulation produces batches
2. Metrics accumulate with delayed labels
3. Drift is detected
4. Retraining is triggered
5. Candidate model is evaluated
6. Promotion occurs if improvement is sufficient

## Running the system
<3 terminal instructions>

## Design decisions
- Why delayed labels
- Why retrain buffer
- Why API-first
- Why readiness checks

## What I’d add next
- Prometheus metrics
- Canary deployments
- Auth

-

# Model Monitor

A production-style **model monitoring & retraining system** that simulates real-world ML inference, drift detection, metrics tracking, and automated retraining with promotion logic.

This project is intentionally designed to mirror how **real ML platforms** (internal tools at companies like Stripe, Uber, or Canva) monitor models in production — not toy notebooks.

---

## ✨ What This Project Demonstrates

* **Online inference simulation** with delayed labels
* **Feature drift detection** using Population Stability Index (PSI)
* **Rolling metrics tracking** (accuracy, F1, confidence)
* **Persistent metrics history** for auditability
* **Automated retraining triggers** based on drift + performance
* **Model comparison & promotion logic**
* **Production-style FastAPI service** with health & readiness probes
* **Clear separation of concerns** (inference, monitoring, training, API)

This is a *systems* project, not just a model.

---

## 🧠 High-Level Architecture

```
                    ┌─────────────────────┐
                    │   run_simulation.py │
                    │  (streaming driver) │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │     Predictor       │
                    │  (inference layer)  │
                    └─────────┬───────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │              Monitoring                 │
        │                                         │
        │  MetricTracker → MetricsHistory (disk)  │
        │  DriftMonitor (PSI)                     │
        │  RetrainController                      │
        └─────────┬───────────────────────────────┘
                  │ retrain decision
                  ▼
        ┌─────────────────────────────────────────┐
        │             Retraining                  │
        │                                         │
        │  RetrainBuffer → train_model()          │
        │  validate_model()                       │
        │  compare_models()                       │
        └─────────┬───────────────────────────────┘
                  │ promote
                  ▼
        ┌─────────────────────────────────────────┐
        │           Model Registry                │
        │  models/current.pkl                     │
        │  models/candidate.pkl                   │
        └─────────────────────────────────────────┘

                              ▲
                              │
                    ┌─────────────────────┐
                    │       FastAPI       │
                    │  /metrics /health  │
                    └─────────────────────┘
```

---

## 📂 Repository Structure


---

## 🔁 Streaming & Retraining Logic

1. Batches are generated and passed to the predictor
2. Labels arrive **with delay** (simulating real systems)
3. Metrics are updated once labels resolve
4. PSI drift is computed on a rolling window
5. Retraining is triggered if:

   * Drift exceeds threshold **and**
   * Accuracy falls below minimum
6. Candidate model is trained and evaluated
7. Candidate is promoted only if improvement exceeds threshold

All decisions are **logged and persisted**.

---

## 🌡 Health & Readiness

* `GET /health` → process liveness
* `GET /ready` → model artifact existence & loadability

These endpoints are designed to work with container orchestration systems.

---

## 📊 Metrics API

Example response from `/metrics/latest`:

```json
{
  "timestamp": 1734829012.2,
  "batch_id": "batch_17_a3f1c9",
  "accuracy": 0.81,
  "f1": 0.79,
  "avg_confidence": 0.72,
  "drift_score": 0.26,
  "action": "retrain",
  "reason": "psi_threshold_exceeded"
}
```

---

## 🖥 Optional UI (Phase 9B)

A minimal Streamlit dashboard can be layered on top of the API to visualize:

* Accuracy / F1 over time
* Drift score trend
* Retraining events

This UI is intentionally thin — the **core value is the backend system**.

---

## 🚀 Why This Project Matters

Most ML portfolios stop at training a model.

This project shows:

* you understand **model lifecycle management**
* you can build **robust, inspectable systems**
* you think like an ML engineer, not a notebook user

This is the kind of project that gets *follow-up questions* in interviews.

---

## ▶️ How to Run

```bash
python training/train.py
python scripts/run_simulation.py
uvicorn api.main:app --reload
```

---

## 🧩 Future Extensions (Optional)

* Model versioning metadata
* Canary deployments
* Alerting hooks (Slack / email)
* Offline evaluation reports

---

Built to be **clear, realistic, and production-minded**.





README V2
# Model Monitor

A production-style model monitoring and retraining system that simulates **real-world ML inference under drift**, tracks performance, triggers retraining, and safely promotes new models — all with clear separation of concerns and auditability.

This project is intentionally designed as a **portfolio-grade MLOps / Applied ML Engineering system**, not a toy demo.

---

## Why this project exists

Most ML demos stop at training a model.

Real systems must answer harder questions:

* What happens *after* deployment?
* How do we detect drift?
* When should we retrain?
* How do we promote models safely?
* How do we observe and audit decisions?

**Model Monitor** answers those questions end‑to‑end.

---

## High-level architecture

```
┌────────────┐
│  Data      │  (streaming / delayed labels)
└─────┬──────┘
      ↓
┌────────────┐
│ Predictor  │  inference + monitoring
└─────┬──────┘
      ↓
┌────────────┐
│ Decision   │  policy engine
│ Engine     │
└─────┬──────┘
      ↓
┌────────────┐
│ Retrain    │  buffer + pipeline
│ Pipeline   │
└─────┬──────┘
      ↓
┌────────────┐
│ Model      │  store + promotion
│ Store      │
└────────────┘
```

Each box is deliberately isolated. No layer “knows too much”.

---

## Core concepts

### 1. Predictor (inference layer)

* Loads the active production model
* Enforces feature schema strictly
* Computes:

  * accuracy
  * F1
  * confidence
  * drift score (PSI)
* Emits a **Decision** object per batch
* Writes metrics and prediction logs for auditability

The predictor is *stateful* only where necessary (cooldowns, batch index).

---

### 2. Decision Engine (policy layer)

The decision engine converts signals into actions:

* `none` – system healthy
* `retrain` – sustained performance degradation
* `reject` – severe feature drift

Rules are fully configurable via YAML.

Importantly:

> The decision engine contains **policy only**, not metrics computation.

---

### 3. Retrain Buffer

In production, labels arrive late.

The retrain buffer:

* Accumulates labeled batches
* Enforces a minimum sample threshold
* Produces a clean training dataframe when ready

This prevents noisy, premature retraining.

---

### 4. Retrain Pipeline

The retrain pipeline executes **exactly one retrain request**:

* Train candidate model
* Evaluate candidate vs current
* Decide promotion based on minimum F1 improvement
* Persist and promote via the model store

It does **not** decide *when* to retrain — that is handled upstream.

---

### 5. Model Store

The model store handles **all persistence concerns**:

* Active model loading
* Candidate saving
* Promotion
* Archiving old models with timestamps
* Writing promotion metadata for audit

This allows safe rollback and historical inspection.

---

### 6. Runtime orchestration

The orchestration loop (e.g. `run_simulation.py`) owns:

* Predictor lifecycle
* Retrain triggering
* Automatic model reload after promotion

This mirrors real streaming / batch systems.

---

## Repository structure

```
model_monitor/
├── api/            # FastAPI endpoints
├── config/         # YAML + typed settings
├── core/           # decision logic
├── data/           # raw / processed / streams
├── inference/      # predictor
├── monitoring/     # drift, metrics, thresholds
├── models/         # active, candidate, archive
├── scripts/        # simulation entrypoints
├── storage/        # model persistence
├── tests/          # unit + integration tests
├── training/       # train / eval / promote
├── ui/             # Streamlit dashboard
├── utils/          # shared utilities
```

---

## Running the system

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train an initial model

```bash
python training/train.py
```

### 3. Run the streaming simulation

```bash
python scripts/run_simulation.py
```

This will:

* Stream batches
* Detect drift
* Trigger retraining
* Promote models
* Reload predictor automatically

---

## Dashboard

A Streamlit dashboard is included:

```bash
streamlit run ui/streamlit_app.py
```

It displays:

* Service health
* Latest metrics
* Metric history over time

---

## Testing philosophy

This project uses **three layers of tests**:

1. Unit tests

   * Decision engine rules
   * Promotion logic

2. Inference tests

   * Golden path prediction
   * Failure paths (schema, missing model)

3. Integration tests

   * Retrain → promotion → reload lifecycle

Tests focus on *behavior*, not implementation details.

---

## What this project demonstrates

* Production ML thinking
* Clean architecture
* Explicit model lifecycle management
* Monitoring-driven retraining
* Auditability and reproducibility

This is the kind of system you build **after** your first real deployment.

---

## Future extensions (intentional)

* Async retraining workers
* Model rollback API
* SHAP-based explanations
* Feature-level drift dashboards
* Canary deployments

---

## Final note

This repository is designed to be **readable, realistic, and honest**.

There are no magic frameworks, no hidden globals, and no black boxes — just clear engineering decisions.

# Model Monitor (Datacheck)

A production‑style **model monitoring and retraining system** designed to look and feel like something you would actually ship.

This project focuses on:

* Detecting data drift and quality issues
* Deciding *when* retraining should happen
* Safely retraining and promoting models
* Keeping runtime prediction services stable

It is intentionally opinionated, pragmatic, and minimal — no toy abstractions, no academic fluff.

---

## Why this project exists

Most ML portfolios stop at:

* training a model
* evaluating it once
* saving it to disk

Real systems don’t fail because the model was bad — they fail because **nobody noticed it quietly becoming wrong**.

This project demonstrates:

* how monitoring feeds retraining
* how retraining feeds promotion
* how promotion safely feeds production

If you were hired to own an ML system long‑term, this is the part you’d actually be responsible for.

---

## High‑level architecture

```
┌────────────┐
│  Incoming  │
│   Events   │
└─────┬──────┘
      │
      ▼
┌────────────┐
│ Validation │  → schema, ranges, nulls
└─────┬──────┘
      │
      ▼
┌────────────┐
│ Drift &    │  → PSI, distribution checks
│ Trust Scoring│
└─────┬──────┘
      │
      ▼
┌────────────┐
│ Simulation │  → decides retrain trigger
└─────┬──────┘
      │
      ▼
┌────────────┐
│ Retrain    │  → train → evaluate → promote
│ Pipeline   │
└─────┬──────┘
      │
      ▼
┌────────────┐
│ Model Store│  → versioned, rollback‑safe
└────────────┘
```

Key design choice:

> **Monitoring never directly retrains.**
> It only *requests* retraining.

---

## Core components

### Event ingestion

* Events represent real model inputs
* Validated before any monitoring logic runs
* Stored for drift analysis and retraining

Design goal: bad data should fail *fast* and *loud*.

---

### Drift & trust scoring

* PSI‑based distribution drift
* Feature‑level checks
* Aggregate trust score

This keeps decisions explainable:

> *“Retraining happened because feature X drifted by Y%.”*

---

### Retrain pipeline

The retrain pipeline is intentionally **dumb**.

It:

* trains a candidate model
* evaluates it
* compares against the current model
* promotes only if improvement is real

It does **not**:

* decide *when* to retrain
* know about events or queues

That separation keeps it testable and safe.

---

### Model promotion & storage

* Candidate models are stored separately
* Promotion is explicit and reversible
* No in‑place overwrites

This mirrors real production systems where rollback matters.

---

### Predictor runtime

The prediction service:

* loads the **current** model
* does not care about retraining
* can reload models without restarting

This prevents outages during promotion.

---

## Automatic predictor reload (design note)

The predictor watches the model store for changes:

* model version file
* symlink change
* metadata timestamp

On change:

* reloads the model into memory
* swaps references atomically

Why this matters:

* no service restart
* no dropped requests
* clean separation of concerns

---

## Testing strategy

This project focuses on **integration tests**, not just unit tests.

Why?

* Individual functions are simple
* Failure happens at boundaries

Covered scenarios:

* retraining with empty data
* promotion blocked on insufficient improvement
* promotion success path
* predictor reload after promotion

---

## What this project intentionally avoids

* AutoML
* Over‑engineering
* Abstract base classes everywhere
* "one framework to rule them all"

Everything here is explicit and debuggable.

---

## How to run

```bash
pip install -r requirements.txt
pytest
python run_simulation.py
```

---

## What this demonstrates to employers

* You understand ML *systems*, not just models
* You know where real failures happen
* You design for safety, rollback, and observability
* You write code that looks maintainable

This is the kind of project that starts real technical conversations.

---

## Future extensions

* Shadow deployments
* Canary promotion
* Online performance metrics
* Feature attribution drift

The core architecture already supports these.

---

## Final note

This repo is not meant to be flashy.

It is meant to look like something that already survived its first on‑call rotation.


---

### Core Design Principles

#### 1. Separation of Policy and Execution
- `DecisionEngine` is **pure** and deterministic
- `DecisionExecutor` handles concurrency, retries, and execution state
- Side effects are isolated in `DefaultModelActionExecutor`

This makes the system testable and auditable.

---

#### 2. Explicit Invariants
System invariants are enforced at aggregation boundaries:
- metrics are finite
- trust scores are bounded
- batch counts are monotonic

Invariant violations fail fast and loudly, preventing silent corruption.

---

#### 3. Idempotent Retraining
Retraining decisions are guarded by:
- evidence thresholds
- retrain keys
- async execution locks
- optional dry-run mode

This prevents duplicate or runaway retraining jobs.

---

#### 4. Auditability
All operational decisions are:
- recorded in-memory for short-term context
- persisted via an append-only decision log
- queryable via analytics interfaces

No decision is executed without being explainable after the fact.

---

#### 5. Production-First Testing
Tests validate **contracts**, not implementations:
- retrain idempotency
- dry-run behavior
- snapshot state transitions

This keeps tests stable while allowing internal refactors.
