"""Health and readiness endpoints for the FastAPI app."""
from __future__ import annotations

from fastapi import APIRouter

from model_monitor.storage.model_store import ModelStore

router = APIRouter(tags=["health"])

# Lazy singleton - model store is created on first request so importing
# this module does not create the models/ directory as a side effect.
_model_store: ModelStore | None = None


def _get_model_store() -> ModelStore:
    global _model_store
    if _model_store is None:
        _model_store = ModelStore()
    return _model_store


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def readiness() -> dict[str, object]:
    try:
        _get_model_store().load_current()
    except FileNotFoundError:
        return {"ready": False, "reason": "model_not_found"}
    except Exception as exc:
        return {
            "ready": False,
            "reason": "model_load_failed",
            "error": str(exc),
        }

    return {"ready": True}
