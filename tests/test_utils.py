"""
Tests for utils/io.py and utils/time.py.

utils/io.load_yaml is the single YAML loading function used across the project.
utils/time provides UTC timestamp helpers.
"""
from __future__ import annotations

import textwrap
import time as _time
from datetime import timezone
from pathlib import Path

import pytest

from model_monitor.utils.io import ensure_dir, load_yaml
from model_monitor.utils.time import utc_iso, utc_now

# ---------------------------------------------------------------------------
# load_yaml
# ---------------------------------------------------------------------------

def test_load_yaml_returns_dict_for_mapping(tmp_path: Path) -> None:
    f = tmp_path / "config.yaml"
    f.write_text("key: value\nnumber: 42\n")
    result = load_yaml(f)
    assert result == {"key": "value", "number": 42}


def test_load_yaml_accepts_string_path(tmp_path: Path) -> None:
    f = tmp_path / "config.yaml"
    f.write_text("a: 1\n")
    result = load_yaml(str(f))
    assert result["a"] == 1


def test_load_yaml_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nonexistent.yaml")


def test_load_yaml_returns_none_for_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.yaml"
    f.write_text("")
    result = load_yaml(f)
    assert result is None


def test_load_yaml_nested_structure(tmp_path: Path) -> None:
    f = tmp_path / "nested.yaml"
    f.write_text(textwrap.dedent("""
        drift:
          psi_threshold: 0.2
          window: 500
    """))
    result = load_yaml(f)
    assert result["drift"]["psi_threshold"] == 0.2


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------

def test_ensure_dir_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()
    ensure_dir(target)
    assert target.is_dir()


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "dir"
    ensure_dir(target)
    ensure_dir(target)  # second call must not raise
    assert target.is_dir()


def test_ensure_dir_returns_path(tmp_path: Path) -> None:
    target = tmp_path / "returned"
    result = ensure_dir(target)
    assert result == target


# ---------------------------------------------------------------------------
# utc_now / utc_iso
# ---------------------------------------------------------------------------

def test_utc_now_is_timezone_aware() -> None:
    dt = utc_now()
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc


def test_utc_iso_is_string() -> None:
    result = utc_iso()
    assert isinstance(result, str)
    assert "T" in result  # ISO 8601 format


def test_utc_now_is_recent() -> None:
    """Timestamp must be within a second of the current time."""
    before = _time.time()
    dt = utc_now()
    after = _time.time()
    ts = dt.timestamp()
    assert before - 1 <= ts <= after + 1
