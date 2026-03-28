.PHONY: install run sim test lint

install:
<<<<<<< HEAD
	pip install -e ".[dev]"
=======
	pip install -e .
>>>>>>> main

run:
	uvicorn model_monitor.api.main:app --reload --port 8000

sim:
	python -m model_monitor.scripts.simulation_loop

test:
	pytest tests/ -v

lint:
<<<<<<< HEAD
	mypy src/ --strict
=======
>>>>>>> main
	ruff check src/