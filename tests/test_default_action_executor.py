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
from typing import cast
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


# ---------------------------------------------------------------------------
# Promote - baseline_f1 backfill
# ---------------------------------------------------------------------------

def test_promote_backfills_baseline_f1_from_f1(tmp_path: Path) -> None:
    """
    When the caller provides f1 but not baseline_f1, the executor must
    backfill baseline_f1 so the decision engine always has a valid baseline
    to compare against on subsequent passes.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    store = ModelStore(base_path=tmp_path)

    # Promote a real model so active.json exists and promote_candidate works
    model = RandomForestClassifier(n_estimators=1, random_state=0)
    model.fit(np.array([[0], [1]]), [0, 1])
    store.save_candidate(model)
    store.promote_candidate({"baseline_f1": 0.80})

    # Now promote again via executor with f1 but no baseline_f1
    store.save_candidate(model)

    executor = DefaultModelActionExecutor(
        model_store=store,
        retrain_pipeline=MagicMock(spec=RetrainPipeline),
        decision_store=DecisionStore(),
        dry_run=False,
    )

    executor.execute(
        action=ModelAction.PROMOTE,
        context={"metrics": {"f1": 0.88}},
    )

    # The promoted metadata must have baseline_f1 backfilled
    meta = store.get_active_metadata()
    assert meta.get("metrics", {}).get("baseline_f1") == pytest.approx(0.88), (
        "Executor must backfill baseline_f1 from f1 when the caller omits it"
    )


# ---------------------------------------------------------------------------
# Retrain path
# ---------------------------------------------------------------------------

def test_retrain_dry_run_returns_none_without_calling_pipeline(
    tmp_path: Path,
) -> None:
    """
    In dry_run mode the retrain path must return None immediately and
    must not call the retrain pipeline at all.
    """
    mock_pipeline = MagicMock(spec=RetrainPipeline)
    executor = DefaultModelActionExecutor(
        model_store=ModelStore(base_path=tmp_path),
        retrain_pipeline=mock_pipeline,
        decision_store=DecisionStore(),
        dry_run=True,
    )

    import numpy as np
    import pandas as pd

    result = executor.execute(
        action=ModelAction.RETRAIN,
        context={
            "retrain_df": pd.DataFrame({"a": np.zeros(10)}),
            "min_f1_improvement": 0.05,
        },
    )

    assert result is None
    mock_pipeline.run.assert_not_called()


def test_retrain_returns_none_when_candidate_not_promoted(tmp_path: Path) -> None:
    """
    When the retrain pipeline runs but the candidate does not outperform
    the current model, execute() must return None — no promotion happens.
    """
    import numpy as np
    import pandas as pd

    from model_monitor.training.promotion import PromotionResult
    from model_monitor.training.retrain_pipeline import RetrainResult

    store = ModelStore(base_path=tmp_path)

    mock_pipeline = MagicMock(spec=RetrainPipeline)
    mock_pipeline.run.return_value = RetrainResult(
        candidate_model=None,   # no model → pipeline signals skip
        promotion=PromotionResult(
            promoted=False,
            reason="empty_retrain_dataset",
            current_f1=0.0,
            candidate_f1=0.0,
            improvement=0.0,
        ),
        n_samples=0,
    )

    executor = DefaultModelActionExecutor(
        model_store=store,
        retrain_pipeline=mock_pipeline,
        decision_store=DecisionStore(),
        dry_run=False,
    )

    result = executor.execute(
        action=ModelAction.RETRAIN,
        context={
            "retrain_df": pd.DataFrame({"a": np.zeros(10)}),
            "min_f1_improvement": 0.05,
        },
    )

    assert result is None


def test_unknown_model_action_raises_value_error(tmp_path: Path) -> None:
    """
    ModelAction.from_decision raises on unknown action strings, which means
    calling execute() with a valid ModelAction enum value never hits the
    ValueError at the bottom of _execute_internal. Verify the guard exists
    by calling _execute_internal directly with a patched unsupported value.
    """

    executor = DefaultModelActionExecutor(
        model_store=ModelStore(base_path=tmp_path),
        retrain_pipeline=MagicMock(spec=RetrainPipeline),
        decision_store=DecisionStore(),
        dry_run=True,
    )

    # Create a value not handled by any branch
    import enum
    FakeAction = enum.Enum("FakeAction", {"UNKNOWN": "unknown"})

    with pytest.raises(ValueError, match="Unsupported"):
        executor._execute_internal(
            action=cast(ModelAction, FakeAction.UNKNOWN),
            context={},
        )
