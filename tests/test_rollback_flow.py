from __future__ import annotations

from pathlib import Path

from model_monitor.storage.model_store import ModelStore


def test_rollback_switches_active_model(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)

    # First model
    store.save_candidate({"model": "v1"})
    v1 = store.promote_candidate()

    # Second model
    store.save_candidate({"model": "v2"})
    v2 = store.promote_candidate()

    assert store.get_active_version() == v2

    # Roll back to v1
    rolled_back = store.rollback(v1)

    assert rolled_back == v1
    assert store.get_active_version() == v1
