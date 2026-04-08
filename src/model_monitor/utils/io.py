"""File I/O utilities: YAML loading and directory creation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> Any:
    """
    Load a YAML file and return its parsed contents.

    Raises:
        FileNotFoundError: if the file does not exist.
        yaml.YAMLError: if the file is not valid YAML.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"YAML file not found: {resolved}")
    with resolved.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    """Create directory and all parents. Return the path unchanged."""
    path.mkdir(parents=True, exist_ok=True)
    return path
