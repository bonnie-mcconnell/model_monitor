"""
Tests for SnapshotStore - the write-ahead log for decision execution.

The critical invariant: a snapshot is written to the DB before execution
begins so that a crash between write and completion leaves a recoverable
record. On restart, is_retrain_key_known() returns True and the duplicate
retrain is suppressed.

These tests verify the store in isolation. The executor integration is
tested in test_decision_executor.py.
"""
from __future__ import annotations

import time

import pytest

from model_monitor.core.decision_snapshot import DecisionSnapshot
from model_monitor.storage.snapshot_store import SnapshotStore


def _make_snapshot(
    decision_id: str = "d-001",
    action: str = "retrain",
    status: str = "pending",
    retrain_key: str | None = None,
) -> DecisionSnapshot:
    return DecisionSnapshot(
        decision_id=decision_id,
        action=action,  # type: ignore[arg-type]
        timestamp=time.time(),
        status=status,
        retrain_key=retrain_key,
    )


# ---------------------------------------------------------------------------
# write / update_status
# ---------------------------------------------------------------------------

def test_write_persists_snapshot_and_is_retrievable() -> None:
    """
    write() must persist the snapshot so it appears in is_retrain_key_known()
    when a retrain_key is set. A write that silently discards the record
    would let crash recovery fail without any visible error.
    """
    store = SnapshotStore()
    snapshot = _make_snapshot(
        decision_id="persist-001",
        retrain_key="sha256-abc",
    )
    store.write(snapshot)
    assert store.is_retrain_key_known("sha256-abc")


def test_write_is_idempotent() -> None:
    """
    A second write for the same decision_id must be silently ignored.
    Idempotency means callers can safely retry on network failure without
    creating duplicate records that would double-count retrain keys.
    """
    store = SnapshotStore()
    snap = _make_snapshot(
        decision_id="idem-002",
        status="pending",
        retrain_key="sha256-idem",
    )
    store.write(snap)
    store.write(snap)  # must not raise or duplicate

    pending = store.pending_retrains()
    matching = [r for r in pending if r.decision_id == "idem-002"]
    assert len(matching) == 1, (
        f"Expected exactly 1 row for idempotent write, got {len(matching)}"
    )


def test_update_status_reflects_new_status() -> None:
    store = SnapshotStore()
    snap = _make_snapshot(decision_id="upd-001", status="pending")
    store.write(snap)

    snap.status = "executed"
    store.update_status(snap)

    # Verify via pending_retrains - executed snapshots must not appear
    pending = store.pending_retrains()
    pending_ids = [r.decision_id for r in pending]
    assert "upd-001" not in pending_ids


def test_update_status_raises_for_unknown_snapshot() -> None:
    """update_status before write is a programming error - must raise."""
    store = SnapshotStore()
    snap = _make_snapshot(decision_id="never-written")
    with pytest.raises(RuntimeError, match="write\\(\\) must be called before"):
        store.update_status(snap)


# ---------------------------------------------------------------------------
# is_retrain_key_known - the crash recovery predicate
# ---------------------------------------------------------------------------

def test_unknown_retrain_key_returns_false() -> None:
    store = SnapshotStore()
    assert store.is_retrain_key_known("nonexistent-key-xyz") is False


def test_known_retrain_key_returns_true() -> None:
    """
    After a snapshot with a retrain_key is written, is_retrain_key_known
    must return True. This is the mechanism that prevents duplicate retrains
    after a crash: the key is written before execution, so even if the
    process dies during execution, the key is in the DB on restart.
    """
    store = SnapshotStore()
    key = "sha256-abc123-unique-key"
    snap = _make_snapshot(
        decision_id="crash-test-001",
        retrain_key=key,
    )
    store.write(snap)
    assert store.is_retrain_key_known(key) is True


def test_different_retrain_keys_are_independent() -> None:
    store = SnapshotStore()
    key_a = "key-aaa-unique"
    key_b = "key-bbb-unique"

    snap = _make_snapshot(decision_id="crash-test-002", retrain_key=key_a)
    store.write(snap)

    assert store.is_retrain_key_known(key_a) is True
    assert store.is_retrain_key_known(key_b) is False


# ---------------------------------------------------------------------------
# pending_retrains - crash recovery enumeration
# ---------------------------------------------------------------------------

def test_pending_retrains_returns_only_pending_retrain_snapshots() -> None:
    """
    pending_retrains() is called on restart. It must return only snapshots
    where action='retrain' and status='pending'. Completed retrains and
    non-retrain actions must not appear.
    """
    store = SnapshotStore()
    unique = f"{time.time():.0f}"

    pending_snap = _make_snapshot(
        decision_id=f"pending-{unique}",
        action="retrain",
        status="pending",
        retrain_key=f"key-pending-{unique}",
    )
    store.write(pending_snap)

    executed_snap = _make_snapshot(
        decision_id=f"executed-{unique}",
        action="retrain",
        status="executed",
        retrain_key=f"key-executed-{unique}",
    )
    store.write(executed_snap)
    executed_snap.status = "executed"
    store.update_status(executed_snap)

    promote_snap = _make_snapshot(
        decision_id=f"promote-{unique}",
        action="promote",
        status="pending",
    )
    store.write(promote_snap)

    pending = store.pending_retrains()
    pending_ids = {r.decision_id for r in pending}

    assert f"pending-{unique}" in pending_ids
    assert f"executed-{unique}" not in pending_ids
    assert f"promote-{unique}" not in pending_ids
