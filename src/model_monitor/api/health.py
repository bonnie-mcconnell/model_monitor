from fastapi import APIRouter

from model_monitor.storage.model_store import ModelStore

router = APIRouter(tags=["health"])

model_store = ModelStore()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def readiness():
    try:
        model_store.load_current()
    except FileNotFoundError:
        return {"ready": False, "reason": "model_not_found"}
    except Exception as e:
        return {
            "ready": False,
            "reason": "model_load_failed",
            "error": str(e),
        }

    return {"ready": True}
