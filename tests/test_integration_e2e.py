"""End-to-end integration test: the system detects drift and self-heals.

This test is the most important one in the suite.  It proves the claim made
in the README: "clone the repo, run make train && make sim, watch the system
detect drift and retrain a new model automatically."

What it verifies
----------------
1. PSI stays near zero for the first ``DRIFT_AT - 1`` batches (stable phase).
2. PSI rises after batch ``DRIFT_AT`` (drift detected).
3. A ``retrain`` action fires after the drift-induced F1 degradation.
4. A new model is promoted - ``model_store.list_versions()`` grows and
   ``make sim`` ends with at least one ``↑ new model promoted`` print.
5. The simulation completes without raising.

Why not just test every component in isolation?
-----------------------------------------------
Unit tests verify that each component does the right thing given its inputs.
This test verifies that the components are *wired together correctly* and
that the end-to-end data flow produces the expected operational behaviour.
It is the difference between "the engine, gearbox, and wheels all work" and
"the car drives."

Runtime: ~4 seconds (60 batches × 20 ms sleep).
"""

from __future__ import annotations

import io
import os
import re
from collections.abc import Iterator
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from model_monitor.config.settings import load_config
from model_monitor.scripts.simulation_loop import simulate_stream
from model_monitor.storage.model_store import ModelStore
from model_monitor.training.train import main as train_main

# Register the custom mark so pytest doesn't emit warnings
pytestmark = pytest.mark.slow


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so regex patterns match cleanly."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture(scope="module")
def trained_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Bootstrap a trained model in an isolated temp directory.

    Scope is ``module`` so the 2-second training step runs once and is
    shared across every test in this file.

    Returns the base data directory for the trained environment.

    Cleanup restores the ``model_monitor`` logger's ``propagate`` flag so
    that ``caplog`` fixtures in subsequent test modules receive log records
    correctly.  ``_configure_sim_logging()`` sets ``propagate=False`` to
    prevent simulation noise from reaching pytest's root handler; without
    cleanup this leaks across the test session.
    """
    import logging as _logging

    base = tmp_path_factory.mktemp("e2e_env")
    (base / "data" / "reference").mkdir(parents=True)
    (base / "data" / "metrics").mkdir(parents=True)
    (base / "data" / "logs").mkdir(parents=True)
    (base / "models" / "archive").mkdir(parents=True)

    prev_cwd = os.getcwd()
    mm_log = _logging.getLogger("model_monitor")
    prev_propagate = mm_log.propagate

    os.chdir(base)
    try:
        train_main()
    finally:
        os.chdir(prev_cwd)

    yield base

    # Restore logger state so caplog works in tests that run after this module.
    mm_log.propagate = prev_propagate
    for handler in list(mm_log.handlers):
        mm_log.removeHandler(handler)
        handler.close()


def test_system_self_heals_under_drift(trained_env: Path) -> None:
    """Full pipeline: stable → drift → retrain → promote.

    This is the single most important test: it asserts that the complete
    monitoring and recovery loop works end-to-end, not just in isolation.
    """
    prev_cwd = os.getcwd()
    os.chdir(trained_env)
    try:
        cfg = load_config()
        # Explicit base_path so list_versions() resolves correctly even after
        # chdir is restored.  ModelStore uses Path(".") by default which is
        # relative - it would point at prev_cwd after os.chdir(prev_cwd).
        store = ModelStore(base_path=trained_env)
        versions_before = len(store.list_versions())

        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            simulate_stream(
                cfg,
                n_batches=60,
                batch_size=200,
                drift_at_batch=30,
                drift_magnitude=2.5,
                sim_drift_window=3,
                data_dir=trained_env / "data" / "reference",
            )

        output = _strip_ansi(stdout_buf.getvalue())

        # 1. Simulation must complete
        assert "simulation finished" in output, (
            "Simulation did not reach the final batch.\n" + output
        )

        # 2. Drift must be detected - at least one "reject" action
        assert "reject" in output, (
            "No 'reject' action - drift was not detected.\n" + output
        )

        # 3. Retrain must fire
        assert "retrain" in output, (
            "No 'retrain' action - retrain was never triggered.\n" + output
        )

        # 4. A new model must be promoted
        assert "new model promoted" in output, (
            "No 'new model promoted' line - the system did not self-heal.\n"
            "This is the core invariant: the monitor must retrain and promote.\n"
            + output
        )

        # 5. Model store must contain more versions than before
        versions_after = len(store.list_versions())
        assert versions_after > versions_before, (
            f"Model store has {versions_after} versions (was {versions_before}) - "
            "no new model was persisted despite output claiming a promotion."
        )

    finally:
        os.chdir(prev_cwd)


def test_stable_batches_produce_no_drift(trained_env: Path) -> None:
    """Before drift is injected PSI should stay near zero.

    Runs 25 batches with no drift injection.  All drift_scores must stay
    below psi_threshold (0.2) because simulation data IS the reference
    distribution - sampled from the same population used to train.
    """
    prev_cwd = os.getcwd()
    os.chdir(trained_env)
    try:
        cfg = load_config()
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            simulate_stream(
                cfg,
                n_batches=25,
                batch_size=200,
                drift_at_batch=999,
                drift_magnitude=0.0,
                sim_drift_window=3,
                data_dir=trained_env / "data" / "reference",
            )

        output = _strip_ansi(stdout_buf.getvalue())

        # Each data row: "       0    0.002    0.994  none"
        drift_scores = [
            float(m.group(1))
            for m in re.finditer(
                r"^\s+\d+\s+([\d.]+)\s+[\d.]+\s+\w", output, re.MULTILINE
            )
        ]
        assert drift_scores, (
            "Could not parse drift scores from simulation output.\n" + output
        )
        high = max(drift_scores)
        assert high < cfg.drift.psi_threshold, (
            f"PSI reached {high:.4f} with no drift injection "
            f"(threshold={cfg.drift.psi_threshold}). "
            "Reference distribution may not match the simulation population."
        )

    finally:
        os.chdir(prev_cwd)
