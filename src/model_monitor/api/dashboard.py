"""FastAPI dashboard routes: metrics, decisions, model actions."""
from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query

from model_monitor.config.settings import load_config
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import DecisionType
from model_monitor.core.model_actions import ModelAction
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.model_store import ModelStore
from model_monitor.training.retrain_pipeline import RetrainPipeline

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


# ---------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------

_ALLOWED_DECISIONS: set[str] = {
    "none",
    "retrain",
    "promote",
    "rollback",
    "reject",
    "system_error",
}


def parse_decision_type(value: str) -> DecisionType:
    if value not in _ALLOWED_DECISIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision action '{value}'",
        )
    return cast(DecisionType, value)


def decision_to_model_action(action: DecisionType) -> ModelAction:
    return {
        "none": ModelAction.NONE,
        "retrain": ModelAction.RETRAIN,
        "promote": ModelAction.PROMOTE,
        "rollback": ModelAction.ROLLBACK,
        "reject": ModelAction.REJECT,
    }.get(action, ModelAction.NONE)


def _orm_to_dict(obj: Any) -> dict[str, Any]:
    """
    Serialise a SQLAlchemy ORM row to a plain dict.

    Using __dict__ directly leaks _sa_instance_state which is not
    JSON-serialisable and exposes SQLAlchemy internals to API consumers.
    This function extracts only the mapped column values.
    """
    return {
        col.key: getattr(obj, col.key)
        for col in obj.__table__.columns
    }


# ---------------------------------------------------------------------
# Stores & services (process-level singletons)
# ---------------------------------------------------------------------

metrics_store = MetricsStore()
summary_store = MetricsSummaryStore()
history_store = MetricsSummaryHistoryStore()
decision_store = DecisionStore()
model_store = ModelStore()
retrain_pipeline = RetrainPipeline(model_store=model_store)

app_config = load_config()
decision_engine = DecisionEngine(config=app_config)


# ---------------------------------------------------------------------
# Metrics (READ-ONLY)
# ---------------------------------------------------------------------

@router.get("/metrics/tail")
def get_metrics_tail(
    limit: int = Query(100, ge=1, le=1000),
) -> list[Any]:
    return metrics_store.tail(limit=limit)


@router.get("/metrics/latest")
def get_latest_metric() -> Any:
    return metrics_store.latest()


@router.get("/metrics")
def list_metrics(
    limit: int = Query(50, ge=1, le=500),
    cursor_ts: float | None = Query(None),
    cursor_id: int | None = Query(None),
    action: str | None = Query(None),
    model: str | None = Query(None),
    min_accuracy: float | None = Query(None, ge=0.0, le=1.0),
    start_ts: float | None = Query(None),
    end_ts: float | None = Query(None),
) -> dict[str, Any]:
    cursor = (
        (cursor_ts, cursor_id)
        if cursor_ts is not None and cursor_id is not None
        else None
    )

    typed_action = parse_decision_type(action) if action else None

    records, next_cursor = metrics_store.list(
        limit=limit,
        cursor=cursor,
        action=typed_action,
        model=model,
        min_accuracy=min_accuracy,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    return {
        "items": records,
        "next_cursor": (
            {"timestamp": next_cursor[0], "id": next_cursor[1]}
            if next_cursor
            else None
        ),
    }


# ---------------------------------------------------------------------
# Aggregated metrics
# ---------------------------------------------------------------------

@router.get("/metrics/summary/{window}")
def get_metrics_summary(window: str) -> dict[str, Any]:
    summary = summary_store.get(window)
    if summary is None:
        raise HTTPException(404, f"No summary for window '{window}'")
    return _orm_to_dict(summary)


@router.get("/metrics/summary/{window}/history")
def get_metrics_summary_history(
    window: str,
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    rows = history_store.list_history(window=window, limit=limit)
    return {
        "window": window,
        "items": [_orm_to_dict(r) for r in rows],
    }


# ---------------------------------------------------------------------
# Decision history
# ---------------------------------------------------------------------

@router.get("/decisions/history")
def get_decision_history(
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    rows = decision_store.tail(limit)
    return [_orm_to_dict(r) for r in reversed(rows)]


# ---------------------------------------------------------------------
# Decision simulation
# ---------------------------------------------------------------------

@router.post("/decisions/simulate")
def simulate_decision() -> dict[str, Any]:
    """
    Simulate what decision the engine would make right now.

    Reads the most recent metric record from the store and the current
    baseline F1 from active.json so the simulation reflects the live
    system state rather than hardcoded example values.
    """
    latest = metrics_store.latest()
    if latest is None:
        return {
            "mode": "simulation",
            "action": "none",
            "reason": "no_metrics_available",
            "would_execute": False,
        }

    active_meta = model_store.get_active_metadata()
    baseline_f1: float = (
        active_meta.get("metrics", {}).get("baseline_f1")
        or latest["f1"]
    )

    recent_actions: list[DecisionType] = []
    for row in decision_store.tail(limit=10):
        try:
            recent_actions.append(parse_decision_type(row.action))
        except HTTPException:
            continue

    decision = decision_engine.decide(
        batch_index=0,
        trust_score=latest.get("avg_confidence", 0.8),
        f1=latest["f1"],
        f1_baseline=baseline_f1,
        drift_score=latest["drift_score"],
        recent_actions=recent_actions,
    )

    return {
        "mode": "simulation",
        "action": decision.action,
        "reason": decision.reason,
        "inputs": {
            "f1": latest["f1"],
            "f1_baseline": baseline_f1,
            "drift_score": latest["drift_score"],
        },
        "would_execute": decision_to_model_action(decision.action) != ModelAction.NONE,
    }


# ---------------------------------------------------------------------
# Manual model actions
# ---------------------------------------------------------------------

@router.post("/models/promote")
def promote_model() -> dict[str, str]:
    version = model_store.promote_candidate({})
    return {"promoted_version": version}


@router.post("/models/rollback")
def rollback_model(version: str) -> dict[str, str]:
    model_store.rollback(version)
    return {"rolled_back_to": version}
