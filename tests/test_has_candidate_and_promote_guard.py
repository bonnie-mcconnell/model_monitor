"""
Tests for ModelStore.has_candidate() and the DecisionEngine candidate_exists guard.

These two additions together close the premature-promote bug:
  - ModelStore.has_candidate() gives callers a reliable way to check whether
    a candidate model is staged without touching the promotion machinery.
  - DecisionEngine.decide(candidate_exists=False) must never return "promote"
    regardless of how many consecutive stable batches have elapsed.

Both properties are safety-critical: a false positive on has_candidate() would
promote garbage; a false negative on the engine guard would fire promote with
FileNotFoundError at every stability window - which is exactly the bug that
was fixed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.storage.model_store import ModelStore

# ---------------------------------------------------------------------------
# ModelStore.has_candidate()
# ---------------------------------------------------------------------------


class TestHasCandidate:
    """has_candidate() must track the presence of the candidate.pkl file."""

    def test_false_on_fresh_store(self, tmp_path: Path) -> None:
        """A brand-new store has no candidate file."""
        store = ModelStore(base_path=tmp_path)
        assert store.has_candidate() is False

    def test_true_after_save_candidate(self, tmp_path: Path) -> None:
        """has_candidate() flips to True once save_candidate() is called."""
        store = ModelStore(base_path=tmp_path)
        store.save_candidate({"weights": [1.0]})
        assert store.has_candidate() is True

    def test_false_after_promote(self, tmp_path: Path) -> None:
        """
        Promoting a candidate removes candidate.pkl from the staging area.
        has_candidate() must return False immediately after promotion so the
        engine does not try to promote a second time in the same window.
        """
        store = ModelStore(base_path=tmp_path)
        store.save_candidate({"weights": [1.0]})
        store.promote_candidate(metrics={"baseline_f1": 0.85})
        assert store.has_candidate() is False

    def test_true_after_second_save(self, tmp_path: Path) -> None:
        """After promotion a new candidate can be staged and detected."""
        store = ModelStore(base_path=tmp_path)
        store.save_candidate({"weights": [1.0]})
        store.promote_candidate(metrics={"baseline_f1": 0.85})
        store.save_candidate({"weights": [2.0]})
        assert store.has_candidate() is True

    def test_reflects_filesystem_directly(self, tmp_path: Path) -> None:
        """
        has_candidate() reads the filesystem, not cached state.
        Manually deleting candidate.pkl must be reflected on the next call.
        This property matters for multi-process deployments where the file
        may be removed by a separate promote process.
        """
        store = ModelStore(base_path=tmp_path)
        store.save_candidate({"weights": [1.0]})
        assert store.has_candidate() is True

        # Simulate external removal
        (tmp_path / "models" / "candidate.pkl").unlink()
        assert store.has_candidate() is False


# ---------------------------------------------------------------------------
# DecisionEngine.decide(candidate_exists=...)
# ---------------------------------------------------------------------------


class TestCandidateExistsGuard:
    """
    The promote rule in decide() must be gated on candidate_exists.

    Without the guard the engine fires "promote" after min_stable_batches
    consecutive "none" actions even when nothing has been staged - causing
    FileNotFoundError in the executor on every stability window.
    """

    @pytest.fixture()
    def engine(self, tmp_path: Path) -> DecisionEngine:
        from model_monitor.config.settings import load_config

        cfg = load_config()
        return DecisionEngine(config=cfg)

    def _stable_actions(self, n: int) -> list[str]:
        return ["none"] * n

    def test_promote_never_fires_without_candidate(
        self, engine: DecisionEngine
    ) -> None:
        """
        After min_stable_batches consecutive "none" actions, decide() must NOT
        return "promote" when candidate_exists=False (the default).

        This is the regression guard for the premature-promote bug.
        """
        n = engine.cfg.retrain.min_stable_batches
        recent = self._stable_actions(n + 5)  # well past the window

        for batch in range(n + 5):
            decision = engine.decide(
                batch_index=batch,
                trust_score=0.95,
                f1=0.88,
                f1_baseline=0.87,
                drift_score=0.001,
                recent_actions=recent,  # type: ignore[arg-type]
                candidate_exists=False,
            )
            assert decision.action != "promote", (
                f"promote fired at batch {batch} with no candidate - "
                "candidate_exists guard is broken"
            )

    def test_promote_fires_with_candidate_and_stable_window(
        self, engine: DecisionEngine
    ) -> None:
        """
        With candidate_exists=True and enough stable batches, "promote" must fire.
        """
        n = engine.cfg.retrain.min_stable_batches
        recent = self._stable_actions(n)

        decision = engine.decide(
            batch_index=n,
            trust_score=0.95,
            f1=0.88,
            f1_baseline=0.87,
            drift_score=0.001,
            recent_actions=recent,  # type: ignore[arg-type]
            candidate_exists=True,
        )
        assert decision.action == "promote"

    def test_promote_still_requires_stable_window(self, engine: DecisionEngine) -> None:
        """
        candidate_exists=True is necessary but not sufficient: the stable-window
        requirement still applies.  A single recent "retrain" breaks the window.
        """
        n = engine.cfg.retrain.min_stable_batches
        # Insert one non-"none" action inside the window
        recent = self._stable_actions(n - 1) + ["retrain"]

        decision = engine.decide(
            batch_index=n,
            trust_score=0.95,
            f1=0.88,
            f1_baseline=0.87,
            drift_score=0.001,
            recent_actions=recent,  # type: ignore[arg-type]
            candidate_exists=True,
        )
        assert decision.action != "promote"

    def test_default_candidate_exists_is_false(self, engine: DecisionEngine) -> None:
        """candidate_exists defaults to False - omitting it is safe."""
        n = engine.cfg.retrain.min_stable_batches
        recent = self._stable_actions(n)

        decision = engine.decide(
            batch_index=n,
            trust_score=0.95,
            f1=0.88,
            f1_baseline=0.87,
            drift_score=0.001,
            recent_actions=recent,  # type: ignore[arg-type]
            # candidate_exists not passed - must default to False
        )
        assert decision.action != "promote"
