"""
Tests for config/settings.py and training/train.py reference stats.

Covers:
- rollback.yaml loading via load_config()
- rollback.yaml fallback when the file is absent
- load_config() accepts explicit rollback_path override
- compute_reference_stats() stores psi_bin_edges
- psi_bin_edges have the correct shape and are monotonically increasing
- Regenerating reference_stats.json round-trips correctly through JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_monitor.config.settings import AppConfig, RollbackConfig, load_config
from model_monitor.training.train import compute_reference_stats

# ---------------------------------------------------------------------------
# rollback.yaml loading
# ---------------------------------------------------------------------------


def test_load_config_loads_rollback_yaml(tmp_path: Path) -> None:
    """
    load_config() must read max_f1_drop from rollback.yaml when supplied.
    Before this fix, RollbackConfig was always constructed with its
    hardcoded default, making the threshold unconfigurable.
    """
    rollback_yaml = tmp_path / "rollback.yaml"
    rollback_yaml.write_text("rollback:\n  max_f1_drop: 0.25\n")

    cfg = load_config(rollback_path=rollback_yaml)

    assert cfg.rollback.max_f1_drop == pytest.approx(0.25)


def test_load_config_rollback_falls_back_when_file_absent(tmp_path: Path) -> None:
    """
    When rollback.yaml does not exist (older deployment), load_config()
    must silently fall back to the hardcoded default (0.15) rather than
    raising FileNotFoundError.

    This preserves backward compatibility for deployments that pre-date
    the rollback.yaml config file.
    """
    missing = tmp_path / "no_rollback.yaml"  # does not exist

    cfg = load_config(rollback_path=missing)

    assert cfg.rollback.max_f1_drop == pytest.approx(0.15)


def test_load_config_bundled_rollback_yaml_loads_cleanly() -> None:
    """
    The rollback.yaml bundled inside the package must load without error
    and produce the documented default value.
    """
    cfg = load_config()

    assert isinstance(cfg.rollback, RollbackConfig)
    assert cfg.rollback.max_f1_drop == pytest.approx(0.15)


def test_rollback_config_in_appconfig_is_correct_type() -> None:
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.rollback, RollbackConfig)


# ---------------------------------------------------------------------------
# compute_reference_stats - psi_bin_edges
# ---------------------------------------------------------------------------


def _make_reference_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))
    df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    df["label"] = rng.integers(0, 2, n)
    return df


def test_compute_reference_stats_includes_psi_bin_edges() -> None:
    """
    compute_reference_stats() must store psi_bin_edges for each feature.
    This is the fix for the architecture claim that 'bin edges are stored
    at training time' - previously only mean/std/hist were stored.
    """
    df = _make_reference_df()
    stats = compute_reference_stats(df)

    for col in ["f0", "f1", "f2", "f3"]:
        assert "psi_bin_edges" in stats[col], f"psi_bin_edges missing for {col}"


def test_psi_bin_edges_have_correct_length() -> None:
    """
    Default psi_bins=10 produces 11 percentile points; np.unique may
    reduce this if values repeat, but must have at least 2 edges for a
    valid histogram.
    """
    df = _make_reference_df()
    stats = compute_reference_stats(df, psi_bins=10)

    for col in ["f0", "f1", "f2", "f3"]:
        edges = stats[col]["psi_bin_edges"]
        assert isinstance(edges, list)
        assert len(edges) >= 2, f"Expected ≥2 edges for {col}, got {len(edges)}"
        assert len(edges) <= 11, f"Expected ≤11 edges for {col}, got {len(edges)}"


def test_psi_bin_edges_are_monotonically_increasing() -> None:
    """
    Bin edges must be strictly increasing - if they are not, np.histogram
    raises a ValueError during drift computation.
    """
    df = _make_reference_df()
    stats = compute_reference_stats(df)

    for col in ["f0", "f1", "f2", "f3"]:
        edges = np.asarray(stats[col]["psi_bin_edges"])
        diffs = np.diff(edges)
        assert np.all(diffs > 0), (
            f"Bin edges for {col} are not strictly increasing: {edges}"
        )


def test_psi_bin_edges_cover_full_reference_range() -> None:
    """
    The first edge must be <= the minimum reference value and the last
    edge must be >= the maximum reference value, so that every reference
    sample falls into a bin.
    """
    df = _make_reference_df()
    stats = compute_reference_stats(df)

    for col in ["f0", "f1", "f2", "f3"]:
        edges = np.asarray(stats[col]["psi_bin_edges"])
        col_min = stats[col]["min"]
        col_max = stats[col]["max"]
        assert edges[0] <= col_min, (
            f"First edge {edges[0]} > column min {col_min} for {col}"
        )
        assert edges[-1] >= col_max, (
            f"Last edge {edges[-1]} < column max {col_max} for {col}"
        )


def test_psi_bin_edges_survive_json_roundtrip() -> None:
    """
    Bin edges are serialised to reference_stats.json; they must survive a
    JSON round-trip without precision loss that would invalidate histogram
    boundaries.
    """
    df = _make_reference_df()
    stats = compute_reference_stats(df)

    serialised = json.dumps(stats)
    recovered = json.loads(serialised)

    for col in ["f0", "f1", "f2", "f3"]:
        original = np.asarray(stats[col]["psi_bin_edges"])
        restored = np.asarray(recovered[col]["psi_bin_edges"])
        np.testing.assert_array_almost_equal(
            original,
            restored,
            decimal=12,
            err_msg=f"psi_bin_edges round-trip failed for {col}",
        )


def test_label_column_excluded_from_reference_stats() -> None:
    df = _make_reference_df()
    stats = compute_reference_stats(df)

    assert "label" not in stats


def test_compute_reference_stats_custom_psi_bins() -> None:
    """
    psi_bins controls how many percentile cuts are made.  psi_bins=5
    should produce ≤6 edges, psi_bins=20 should produce ≤21 edges.
    """
    df = _make_reference_df(n=1000)

    stats_5 = compute_reference_stats(df, psi_bins=5)
    stats_20 = compute_reference_stats(df, psi_bins=20)

    for col in ["f0", "f1"]:
        assert len(stats_5[col]["psi_bin_edges"]) <= 6
        assert len(stats_20[col]["psi_bin_edges"]) <= 21
        # More bins should give equal or more edges
        assert len(stats_20[col]["psi_bin_edges"]) >= len(stats_5[col]["psi_bin_edges"])
