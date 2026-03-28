.PHONY: install run sim test lint

install:
	pip install -e .

run:
	uvicorn model_monitor.api.main:app --reload --port 8000

sim:
	python -m model_monitor.scripts.simulation_loop

test:
	pytest tests/ -v

lint:
	ruff check src/