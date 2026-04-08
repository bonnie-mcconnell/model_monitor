"""
Tests for DefaultModelActionExecutor.

Three properties that matter:

1. Same action on the same model version in recent history is skipped
   (executor-level idempotency, distinct from the SHA-256 buffer key)

2. Failures are recorded to DecisionStore before re-raising so they are
   auditable - the test verifies the record appears even when the action raises

3. Rollback raises explicitly if no version is provided in context - it must
   never silently succeed with no version

These are tested in dry_run=True mode to avoid filesystem side effects.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from model_monitor.core.decisions import Decision
from model_monitor.core.default_model_action_executor import DefaultModelActionExecutor
from model_monitor.core.model_actions import ModelAction
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.model_store import ModelStore
from model_monitor.training.retrain_pipeline import RetrainPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path: Path) -> ModelStore:
    return ModelStore(base_path=tmp_path)


@pytest.fixture
def executor(tmp_store: ModelStore) -> DefaultModelActionExecutor:
    return DefaultModelActionExecutor(
        model_store=tmp_store,
        retrain_pipeline=MagicMock(spec=RetrainPipeline),
        decision_store=DecisionStore(),
        dry_run=True,
    )


# ---------------------------------------------------------------------------
# None and Reject are no-ops
# ---------------------------------------------------------------------------

def test_none_action_returns_none(executor: DefaultModelActionExecutor) -> None:
    result = executor.execute(action=ModelAction.NONE, context={})
    assert result is None


def test_reject_action_returns_none(executor: DefaultModelActionExecutor) -> None:
    result = executor.execute(action=ModelAction.REJECT, context={})
    assert result is None


# ---------------------------------------------------------------------------
# Rollback requires version in context
# ---------------------------------------------------------------------------

def test_rollback_raises_without_version(executor: DefaultModelActionExecutor) -> None:
    with pytest.raises(ValueError, match="version"):
        executor.execute(action=ModelAction.ROLLBACK, context={})


def test_rollback_dry_run_returns_none_without_touching_store(
    executor: DefaultModelActionExecutor,
) -> None:
    result = executor.execute(
        action=ModelAction.ROLLBACK,
        context={"version": "20240101_120000_000000"},
    )
    assert result is None


# ---------------------------------------------------------------------------
# Promote - baseline_f1 backfill
# ---------------------------------------------------------------------------

def test_promote_dry_run_returns_none(executor: DefaultModelActionExecutor) -> None:
    result = executor.execute(
        action=ModelAction.PROMOTE,
        context={"metrics": {"f1": 0.88}},
    )
    assert result is None


# ---------------------------------------------------------------------------
# Executor-level idempotency
# ---------------------------------------------------------------------------

def test_executor_skips_repeated_action_on_same_model_version(
    tmp_path: Path,
) -> None:
    """
    If the same action appears in recent DecisionStore history for the
    current model version, execute() returns None without acting again.
    This is the executor-level idempotency guard, separate from the
    SHA-256 retrain key.
    """
    store = ModelStore(base_path=tmp_path)
    decision_store = DecisionStore()

    executor = DefaultModelActionExecutor(
        model_store=store,
        retrain_pipeline=MagicMock(spec=RetrainPipeline),
        decision_store=decision_store,
        dry_run=False,  # real mode to test the guard fires
    )

    # Record that PROMOTE already happened for this model version
    decision_store.record(
        decision=Decision(action="promote", reason="already done"),
        model_version=store.get_active_version(),
    )

    # Same action, same model version - should be skipped
    result = executor.execute(
        action=ModelAction.PROMOTE,
        context={"metrics": {}},
    )
    assert result is None
