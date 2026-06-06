"""Integration tests for the full retrain pipeline.

Tests in this file exercise the complete retrain path:
  RawDataBuffer → DefaultModelActionExecutor → RetrainPipeline → ModelStore

The critical properties:
- When RawDataBuffer is wired and ready, the executor trains on observed data.
- When RawDataBuffer is absent, the executor falls back to synthetic data
  (logged at WARNING, never raises).
- Reference stats are updated on disk after a successful promotion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_monitor.core.default_model_action_executor import DefaultModelActionExecutor
from model_monitor.monitoring.raw_data_buffer import RawDataBuffer
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.model_store import ModelStore
from model_monitor.training.retrain_pipeline import RetrainPipeline
from model_monitor.training.train import make_dataset, train_model

# Feature names that match make_dataset()'s 10-column output.
_FEATURE_NAMES = [f"f{i}" for i in range(10)]


def _bootstrap_model(model_store: ModelStore) -> None:
    """Train and promote an initial model into model_store."""
    df, _ = make_dataset(n_samples=300, random_state=1)
    model = train_model(df)
    model_store.save_candidate(model)
    model_store.promote_candidate({"baseline_f1": 0.75})


def _make_executor(
    tmp_path: Path,
    *,
    raw_data_buffer: RawDataBuffer | None = None,
) -> tuple[DefaultModelActionExecutor, ModelStore]:
    model_store = ModelStore(base_path=tmp_path)
    decision_store = DecisionStore(db_path=tmp_path / "decisions.db")
    pipeline = RetrainPipeline(model_store=model_store)
    _bootstrap_model(model_store)

    executor = DefaultModelActionExecutor(
        model_store=model_store,
        retrain_pipeline=pipeline,
        decision_store=decision_store,
        raw_data_buffer=raw_data_buffer,
    )
    return executor, model_store


# ---------------------------------------------------------------------------
# Original test - still passes
# ---------------------------------------------------------------------------


def test_retrain_pipeline_promotes_on_improvement(tmp_path: Path) -> None:
    """Full integration: trains initial model, retrains on new data, promotes."""
    model_store = ModelStore(base_path=tmp_path)

    baseline_df = pd.DataFrame(
        {
            "feature_a": [0, 1, 0, 1],
            "feature_b": [1, 1, 0, 0],
            "label": [0, 1, 0, 1],
        }
    )
    initial_model = train_model(baseline_df)
    model_store.save_candidate(initial_model)
    model_store.promote_candidate()

    retrain_df = pd.DataFrame(
        {
            "feature_a": [0, 1, 1, 1, 0, 1],
            "feature_b": [1, 0, 1, 0, 0, 1],
            "label": [0, 1, 1, 1, 0, 1],
        }
    )

    pipeline = RetrainPipeline(model_store=model_store)
    result = pipeline.run(retrain_df, min_f1_improvement=0.01)
    assert result.promotion.promoted is True
    assert result.promotion.candidate_f1 > result.promotion.current_f1
    assert model_store.load_current() is not None


# ---------------------------------------------------------------------------
# RawDataBuffer path - uses _resolve_retrain_dataset directly to avoid
# the execute() idempotency guard, which skips actions already recorded.
# ---------------------------------------------------------------------------


def test_retrain_uses_raw_data_buffer_when_ready(tmp_path: Path) -> None:
    """_resolve_retrain_dataset returns observed data from RawDataBuffer."""
    buf = RawDataBuffer(max_rows=10_000)

    # Populate with 10-feature data matching the bootstrapped model.
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 10))
    y = (X[:, 0] > 0).astype(int)
    buf.add_batch(X, y, _FEATURE_NAMES)

    executor, _ = _make_executor(tmp_path, raw_data_buffer=buf)

    df = executor._resolve_retrain_dataset({"min_samples": 100})

    # Must have feature columns and a label column.
    assert "label" in df.columns
    assert set(_FEATURE_NAMES).issubset(set(df.columns))
    assert len(df) == 300
    # Buffer consumed after resolve.
    assert buf.size() == 0


def test_retrain_falls_back_to_synthetic_when_no_buffer(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When no RawDataBuffer is wired, executor uses synthetic data and warns."""
    executor, _ = _make_executor(tmp_path, raw_data_buffer=None)

    with caplog.at_level(
        logging.WARNING,
        logger="model_monitor.core.default_model_action_executor",
    ):
        df = executor._resolve_retrain_dataset({"min_samples": 100})

    assert "label" in df.columns
    assert any("synthetic_fallback" in r.message for r in caplog.records)


def test_retrain_falls_back_when_buffer_below_min_samples(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Buffer present but below min_samples triggers synthetic fallback."""
    buf = RawDataBuffer(max_rows=10_000)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 10))
    y = (X[:, 0] > 0).astype(int)
    buf.add_batch(X, y, _FEATURE_NAMES)

    executor, _ = _make_executor(tmp_path, raw_data_buffer=buf)

    with caplog.at_level(
        logging.WARNING,
        logger="model_monitor.core.default_model_action_executor",
    ):
        df = executor._resolve_retrain_dataset({"min_samples": 100})

    assert "label" in df.columns
    assert any("synthetic_fallback" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Reference stats update after promotion
# ---------------------------------------------------------------------------


def test_refresh_reference_stats_writes_valid_json(tmp_path: Path) -> None:
    """_refresh_reference_stats writes well-formed reference_stats.json."""
    executor, _ = _make_executor(tmp_path)

    # Build a small retrain DataFrame with the right schema.
    df, _ = make_dataset(n_samples=200, random_state=9)

    ref_path = Path("data/reference/reference_stats.json")
    ref_path.parent.mkdir(parents=True, exist_ok=True)

    # Save any pre-existing content so we can restore after the test.
    original = ref_path.read_text() if ref_path.exists() else None

    try:
        executor._refresh_reference_stats(df)

        assert ref_path.exists(), "reference_stats.json was not written"
        data = json.loads(ref_path.read_text())
        assert isinstance(data, dict)
        # Each feature should have at least a psi_bin_edges key.
        for key in data:
            assert "psi_bin_edges" in data[key], f"Missing psi_bin_edges for {key}"
    finally:
        if original is not None:
            ref_path.write_text(original)
        elif ref_path.exists():
            ref_path.unlink()


def test_refresh_reference_stats_does_not_raise_on_bad_path(tmp_path: Path) -> None:
    """_refresh_reference_stats is best-effort - never propagates exceptions."""
    executor, _ = _make_executor(tmp_path)
    df, _ = make_dataset(n_samples=50, random_state=2)

    # Patch the module-level path to an unwritable location.
    import model_monitor.core.default_model_action_executor as mod

    original_path = mod._REF_STATS_PATH
    mod._REF_STATS_PATH = Path("/proc/no_permission/reference_stats.json")
    try:
        # Must not raise.
        executor._refresh_reference_stats(df)
    finally:
        mod._REF_STATS_PATH = original_path
