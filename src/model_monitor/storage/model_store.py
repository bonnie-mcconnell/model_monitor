from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib # type: ignore


class ModelStore:
    """
    File-based model store with atomic promotion and rollback support.

    Layout:
      models/
        current.pkl
        candidate.pkl
        archive/
          model_<version>.pkl
        active.json
    """

    def __init__(self, base_path: Path | str = "."):
        base = Path(base_path)

        self._models_dir = base / "models"
        self._archive_dir = self._models_dir / "archive"

        self._current_model = self._models_dir / "current.pkl"
        self._candidate_model = self._models_dir / "candidate.pkl"
        self._active_file = self._models_dir / "active.json"

        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Paths (public, read-only)
    # --------------------------------------------------
    @property
    def current(self) -> Path:
        return self._current_model

    @property
    def candidate(self) -> Path:
        return self._candidate_model

    @property
    def archive_dir(self) -> Path:
        return self._archive_dir

    @property
    def active_file(self) -> Path:
        return self._active_file

    # --------------------------------------------------
    # Load / save
    # --------------------------------------------------
    def load_current(self) -> Any:
        if not self._current_model.exists():
            raise FileNotFoundError("No active model found")
        return joblib.load(self._current_model)

    def save_candidate(self, model: Any) -> None:
        joblib.dump(model, self._candidate_model)

    # --------------------------------------------------
    # Promotion
    # --------------------------------------------------
    def promote_candidate(self, metrics: Optional[Dict[str, Any]] = None) -> str:
        if not self._candidate_model.exists():
            raise FileNotFoundError("No candidate model to promote")

        new_version = self._new_version()
        previous_version = self.get_active_version()

        # Archive current model under its *actual* version
        if self._current_model.exists() and previous_version is not None:
            archived = self._archive_dir / f"model_{previous_version}.pkl"
            self._current_model.replace(archived)

        # Promote candidate → current
        self._candidate_model.replace(self._current_model)

        self._write_active(
            version=new_version,
            metadata={
                "previous_version": previous_version,
                "metrics": metrics or {},
                "promoted_at_utc": self._now(),
            },
        )

        return new_version

    # --------------------------------------------------
    # Rollback
    # --------------------------------------------------
    def rollback(self, version: str) -> str:
        """
        Roll back to a specific archived model version.
        """
        archived = self._archive_dir / f"model_{version}.pkl"
        if not archived.exists():
            raise FileNotFoundError(f"No archived model for version '{version}'")

        current_version = self.get_active_version()

        # Archive current model under its real version
        if self._current_model.exists() and current_version is not None:
            self._current_model.replace(
                self._archive_dir / f"model_{current_version}.pkl"
            )

        # Restore archived → current
        archived.replace(self._current_model)

        self._write_active(
            version=version,
            metadata={
                "rollback_from": current_version,
                "rollback_at_utc": self._now(),
            },
        )

        return version

    # --------------------------------------------------
    # Metadata / inspection
    # --------------------------------------------------
    def get_active_version(self) -> Optional[str]:
        if not self._active_file.exists():
            return None
        return json.loads(self._active_file.read_text()).get("version")

    def get_active_metadata(self) -> Dict[str, Any]:
        if not self._active_file.exists():
            return {}
        return json.loads(self._active_file.read_text())

    def list_versions(self) -> List[Dict[str, str]]:
        """
        List archived model versions (newest first).
        """
        versions: List[Dict[str, str]] = []

        for pkl in self._archive_dir.glob("model_*.pkl"):
            version = pkl.stem.replace("model_", "")
            versions.append(
                {
                    "version": version,
                    "path": str(pkl),
                    "created_at": datetime.fromtimestamp(
                        pkl.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )

        return sorted(versions, key=lambda v: v["version"], reverse=True)

    # --------------------------------------------------
    # Internals
    # --------------------------------------------------
    def _write_active(self, *, version: str, metadata: Dict[str, Any]) -> None:
        payload = {
            "version": version,
            "active_model": self._current_model.name,
            **metadata,
        }

        tmp = self._active_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._active_file)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_version() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


# --------------------------------------------------
# Thin convenience layer (module-level API)
# --------------------------------------------------
_default_store = ModelStore()


def load_current() -> Any:
    return _default_store.load_current()


def save_candidate(model: Any) -> None:
    _default_store.save_candidate(model)


def promote_candidate(metrics: Dict[str, Any] | None = None) -> str:
    return _default_store.promote_candidate(metrics)


def rollback(*, version: str) -> str:
    return _default_store.rollback(version)


def get_active_version() -> Optional[str]:
    return _default_store.get_active_version()
