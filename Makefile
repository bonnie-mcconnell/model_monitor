.PHONY: help install run train sim dashboard notebook demo \
        test test-fast coverage lint typecheck fmt pre-commit \
        docker-build docker-up docker-down screenshot

# ── Variables ───────────────────────────────────────────────────────────────

PYTHON   := python
UVICORN  := uvicorn
PYTEST   := pytest
PORT     := 8000
DATASET  := synthetic   # override with: make train DATASET=breast-cancer

# ── Help ────────────────────────────────────────────────────────────────────

help:
	@echo "model-monitor · main"
	@echo ""
	@echo "Setup"
	@echo "  make install       install package + dev dependencies"
	@echo "  make train         train initial model (required before sim or run)"
	@echo ""
	@echo "Development"
	@echo "  make test          run 806 tests (~120 seconds)"
	@echo "  make test-fast     run tests, skip slow integration suite"
	@echo "  make coverage      tests + coverage report (threshold: 80%)"
	@echo "  make lint          ruff check src/ tests/"
	@echo "  make typecheck     mypy src/model_monitor/ tests/"
	@echo "  make fmt           ruff format src/ tests/ (auto-fix)"
	@echo "  make pre-commit    run all pre-commit hooks on every file"
	@echo ""
	@echo "Running"
	@echo "  make demo          train model, then show multi-terminal instructions"
	@echo "  make run           FastAPI server at localhost:8000"
	@echo "  make sim           drift simulation (requires make train first)"
	@echo "  make dashboard     Streamlit UI (requires make run in another terminal)"
	@echo "  make notebook      open the drift simulation notebook"
	@echo ""
	@echo "CLI tools"
	@echo "  make replay        replay decisions from stored history (--from / --to)"
	@echo "  make replay-dry    replay without writing to audit log"
	@echo "  make export        export audit log to CSV (stdout or --output FILE)"
	@echo "  make export-json   export audit log to NDJSON (pipe to jq)"
	@echo ""
	@echo "Docker"
	@echo "  make docker-build  build the Docker image"
	@echo "  make docker-up     start the server in Docker (server only)"
	@echo "  make docker-down   stop the server"
	@echo "  make monitoring    start full stack: server + Prometheus + Grafana"
	@echo "                     Grafana at http://localhost:3000  (admin/admin)"
	@echo "  make monitoring-down  stop the full stack"

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

train:
	$(PYTHON) -m model_monitor.training.train --dataset $(DATASET)

# ── Quality gates ────────────────────────────────────────────────────────────

test:
	$(PYTEST) tests/ -v

test-fast:
	$(PYTEST) tests/ -m "not slow"

coverage:
	$(PYTEST) tests/ \
	    --cov=model_monitor \
	    --cov-report=term-missing \
	    --cov-fail-under=80

lint:
	ruff check src/ tests/

typecheck:
	mypy src/model_monitor/ tests/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

pre-commit:
	pre-commit run --all-files

# ── Running ──────────────────────────────────────────────────────────────────

run:
	$(UVICORN) model_monitor.api.main:app --reload --port $(PORT)

sim:
	$(PYTHON) -m model_monitor.scripts.simulation_loop

replay:
	model-monitor-replay $(ARGS)

replay-dry:
	model-monitor-replay --dry-run $(ARGS)

export:
	model-monitor-export $(ARGS)

export-json:
	model-monitor-export --format json $(ARGS)

dashboard:
	streamlit run src/model_monitor/ui/streamlit_app.py

notebook:
	jupyter notebook notebooks/uci_adult_drift_demo.ipynb

demo:
	@echo ""
	@echo "━━━  model_monitor demo  ━━━"
	@echo ""
	@echo "Step 1 of 4: training initial model..."
	@$(MAKE) -s train
	@echo ""
	@echo "Training complete. Open three more terminals and run:"
	@echo ""
	@echo "  Terminal 2 →  make run        (FastAPI server at localhost:8000)"
	@echo "  Terminal 3 →  make sim        (80-batch drift simulation)"
	@echo "  Terminal 4 →  make dashboard  (Streamlit monitoring UI)"
	@echo ""
	@echo "  API docs  →  http://localhost:8000/docs"
	@echo "  Dashboard →  http://localhost:8501"
	@echo ""

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down

monitoring:
	docker compose --profile monitoring up --build

monitoring-down:
	docker compose --profile monitoring down

# ── Misc ─────────────────────────────────────────────────────────────────────

screenshot:
	@mkdir -p docs
	@command -v svg-term >/dev/null 2>&1 || { echo "Install svg-term: npm i -g svg-term-cli"; exit 1; }
	ttyrec /tmp/sim_rec.ttyrec -e "$(MAKE) sim"
	svg-term --in /tmp/sim_rec.ttyrec --out docs/sim_output.svg --width 100 --height 40
	@echo "Screenshot saved to docs/sim_output.svg"

demo-plot:  ## Generate monitoring dashboard screenshot (docs/demo_plot.png)
	@echo "Generating monitoring dashboard screenshot → docs/demo_plot.png"
	$(PYTHON) src/model_monitor/scripts/demo_plot.py --output docs/demo_plot.png
	@echo "Done - open docs/demo_plot.png"
