from fastapi import APIRouter
from pathlib import Path
import joblib

router = APIRouter(tags=["health"])

MODEL_PATH = Path("models/current.pkl")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def readiness():
    if not MODEL_PATH.exists():
        return {"ready": False, "reason": "model_not_found"}

    try:
        joblib.load(MODEL_PATH)
    except Exception as e:
        return {
            "ready": False,
            "reason": "model_load_failed",
            "error": str(e),
        }

    return {"ready": True}

