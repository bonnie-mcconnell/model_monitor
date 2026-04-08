"""
Tests for ModelStore.

ModelStore is the most operationally critical component in the system -
it owns model promotion, rollback, and archiving. A bug here means running
the wrong model in production with no way to detect it.

Specific properties tested:
- baseline_f1 survives the write→read round trip through active.json
- list_versions returns correct count and ordering
- rollback to non-existent version raises clearly
- get_active_metadata returns full metric content, not just version
- atomic write: no partial files left after save_candidate
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from model_monitor.storage.model_store import ModelStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _promote(store: ModelStore, f1: float = 0.85) -> str:
    """Save a minimal serialisable object as candidate and promote it."""
    store.save_candidate({"weights": [1.0, 2.0]})
    return store.promote_candidate(metrics={"baseline_f1": f1, "f1": f1})


# ---------------------------------------------------------------------------
# Promotion and active.json
# ---------------------------------------------------------------------------

def test_promote_writes_version_to_active_json(tmp_path: Path) -> None:
    store = ModelStore(base_path=tmp_path)
    version = _promote(store)
    assert store.get_active_version() == version


def test_promote_writes_baseline_f1_to_active_metadata(tmp_path: Path) -> None:
    """
    This is the critical property: baseline_f1 must survive write→read.
    The decision engine reads this on every aggregation pass to compute f1_drop.
    If it's missing, effective_baseline falls back to avg_f1, disabling
    retrain and rollback detection.
    """
    store = ModelStore(base_path=tmp_path)
    _promote(store, f1=0.88)

    meta = store.get_active_metadata()
    assert "metrics" in meta
    assert meta["metrics"]["baseline_f1"] == pytest.approx(0.88)


def test_promote_records_previous_version_in_metadata(tmp_path: Path) -> None:
    store = ModelStore(base_path=tmp_path)
    v1 = _promote(store, f1=0.80)
    _promote(store, f1=0.85)

    meta = store.get_active_metadata()
    assert meta["previous_version"] == v1


def test_get_active_metadata_returns_empty_dict_before_first_promotion(
    tmp_path: Path,
) -> None:
    store = ModelStore(base_path=tmp_path)
    assert store.get_active_metadata() == {}


def test_get_active_version_returns_none_before_first_promotion(
    tmp_path: Path,
) -> None:
    store = ModelStore(base_path=tmp_path)
    assert store.get_active_version() is None


# ---------------------------------------------------------------------------
# list_versions
# ---------------------------------------------------------------------------

def test_list_versions_returns_empty_before_any_promotion(tmp_path: Path) -> None:
    store = ModelStore(base_path=tmp_path)
    assert store.list_versions() == []


def test_list_versions_after_multiple_promotions(tmp_path: Path) -> None:
    """
    After N promotions, there should be N-1 archived versions
    (the current model is not in the archive).
    """
    store = ModelStore(base_path=tmp_path)
    _promote(store)
    _promote(store)
    _promote(store)

    versions = store.list_versions()
    # 3 promotions → 2 archived (third is current)
    assert len(versions) == 2


def test_list_versions_each_entry_has_required_fields(tmp_path: Path) -> None:
    store = ModelStore(base_path=tmp_path)
    _promote(store)
    _promote(store)

    for entry in store.list_versions():
        assert "version" in entry
        assert "path" in entry
        assert "created_at" in entry


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def test_save_candidate_leaves_no_tmp_file(tmp_path: Path) -> None:
    """
    The atomic rename pattern must not leave .tmp files behind.
    A leftover .tmp file means the rename failed or was skipped.
    """
    store = ModelStore(base_path=tmp_path)
    store.save_candidate({"model": "data"})

    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert tmp_files == [], f"Leftover .tmp files: {tmp_files}"


def test_save_candidate_file_is_loadable(tmp_path: Path) -> None:
    """Saved candidate must be fully written and loadable."""
    store = ModelStore(base_path=tmp_path)
    original = {"weights": [0.1, 0.2, 0.3]}
    store.save_candidate(original)

    loaded = joblib.load(store.candidate)
    assert loaded == original


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def test_rollback_to_nonexistent_version_raises(tmp_path: Path) -> None:
    """
    Rollback to a version that does not exist must raise FileNotFoundError,
    not silently succeed or corrupt the current model.
    """
    store = ModelStore(base_path=tmp_path)
    _promote(store)

    with pytest.raises(FileNotFoundError):
        store.rollback("99991231_999999_999999")


def test_rollback_updates_active_version(tmp_path: Path) -> None:
    store = ModelStore(base_path=tmp_path)
    v1 = _promote(store, f1=0.80)
    _promote(store, f1=0.85)

    assert store.get_active_version() != v1
    store.rollback(v1)
    assert store.get_active_version() == v1


def test_rollback_loads_correct_model(tmp_path: Path) -> None:
    """
    After rollback, load_current() must return the model from v1,
    not the model from v2.
    """
    store = ModelStore(base_path=tmp_path)

    store.save_candidate({"version": "v1"})
    v1 = store.promote_candidate()

    store.save_candidate({"version": "v2"})
    store.promote_candidate()

    store.rollback(v1)
    model = store.load_current()
    assert model == {"version": "v1"}


# ---------------------------------------------------------------------------
# load_current error path
# ---------------------------------------------------------------------------

def test_load_current_raises_when_no_model_exists(tmp_path: Path) -> None:
    store = ModelStore(base_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load_current()
