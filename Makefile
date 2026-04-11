.PHONY: help install run train sim notebook docker-build docker-up docker-down demo test lint typecheck coverage

help:
	@echo "model-monitor · behavior-monitoring"
	@echo ""
	@echo "  make install    install package + dev dependencies"
	@echo "  make train      train initial model (required before make sim or make run)"
	@echo "  make test       run 320 tests (~17 seconds)"
	@echo "  make coverage   run tests with coverage report (threshold: 88%)"
	@echo "  make lint       ruff check src/ tests/"
	@echo "  make typecheck  mypy src/model_monitor/ tests/"
	@echo "  make demo       behavioral contracts end-to-end demo"
	@echo "  make sim        drift simulation loop"
	@echo "  make notebook   open the drift simulation notebook"
	@echo ""
	@echo "  make docker-build  build the Docker image"
	@echo "  make docker-up     start the server in Docker"
	@echo "  make docker-down   stop the server"
	@echo "  make run        FastAPI server at localhost:8000"

install:
	pip install -e ".[dev]"

train:
	python -m model_monitor.training.train

run:
	uvicorn model_monitor.api.main:app --reload --port 8000

sim:
	python -m model_monitor.scripts.simulation_loop

demo:
	python scripts/demo_contracts.py

test:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=model_monitor --cov-report=term-missing --cov-fail-under=88

lint:
	ruff check src/ tests/

typecheck:
	mypy src/model_monitor/ tests/

notebook:
	jupyter notebook notebooks/drift_simulation.ipynb

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down
