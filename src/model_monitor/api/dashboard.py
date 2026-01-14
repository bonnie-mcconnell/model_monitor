from __future__ import annotations

from typing import Optional, cast

from fastapi import APIRouter, Query, HTTPException

from model_monitor.core.decisions import DecisionType
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.model_actions import ModelAction
from model_monitor.config.settings import load_config

from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.model_store import ModelStore
from model_monitor.training.retrain_pipeline import RetrainPipeline

from model_monitor.storage.models.metrics_summary_history import (
    MetricsSummaryHistoryORM,
)

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
    # explicit cast after validation — no type ignores
    return cast(DecisionType, value)


def decision_to_model_action(action: DecisionType) -> ModelAction:
    return {
        "none": ModelAction.NONE,
        "retrain": ModelAction.RETRAIN,
        "promote": ModelAction.PROMOTE,
        "rollback": ModelAction.ROLLBACK,
        "reject": ModelAction.REJECT,
    }.get(action, ModelAction.NONE)


# ---------------------------------------------------------------------
# Stores & services (process-level singletons)
# ---------------------------------------------------------------------

metrics_store = MetricsStore()
summary_store = MetricsSummaryStore()
decision_store = DecisionStore()
model_store = ModelStore()
retrain_pipeline = RetrainPipeline(model_store=model_store)

app_config = load_config()
decision_engine = DecisionEngine(config=app_config)


# ---------------------------------------------------------------------
# Metrics (READ-ONLY)
# ---------------------------------------------------------------------

@router.get("/metrics/tail")
def get_metrics_tail(limit: int = Query(100, ge=1, le=1000)):
    return metrics_store.tail(limit=limit)


@router.get("/metrics/latest")
def get_latest_metric():
    return metrics_store.latest()


@router.get("/metrics")
def list_metrics(
    limit: int = Query(50, ge=1, le=500),
    cursor_ts: Optional[float] = Query(None),
    cursor_id: Optional[int] = Query(None),
    action: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    min_accuracy: Optional[float] = Query(None, ge=0.0, le=1.0),
    start_ts: Optional[float] = Query(None),
    end_ts: Optional[float] = Query(None),
):
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
def get_metrics_summary(window: str):
    summary = summary_store.get(window)
    if summary is None:
        raise HTTPException(404, f"No summary for window '{window}'")
    return summary


@router.get("/metrics/summary/{window}/history")
def get_metrics_summary_history(
    window: str,
    limit: int = Query(100, ge=1, le=1000),
):
    from model_monitor.storage.metrics_summary_history_store import (
        MetricsSummaryHistoryStore,
    )

    store = MetricsSummaryHistoryStore()

    with store._session_factory() as session:
        rows = (
            session.query(MetricsSummaryHistoryORM)
            .filter(MetricsSummaryHistoryORM.window == window)
            .order_by(MetricsSummaryHistoryORM.timestamp.desc())
            .limit(limit)
            .all()
        )

    return {
        "window": window,
        "items": [r.__dict__ for r in reversed(rows)],
    }


# ---------------------------------------------------------------------
# Decision history
# ---------------------------------------------------------------------

@router.get("/decisions/history")
def get_decision_history(limit: int = Query(100, ge=1, le=1000)):
    rows = decision_store.tail(limit)
    return [r.__dict__ for r in reversed(rows)]


# ---------------------------------------------------------------------
# Phase IV.1 — Decision simulation ONLY
# ---------------------------------------------------------------------

@router.post("/decisions/simulate")
def simulate_decision():
    recent_actions: list[DecisionType] = []

    for row in decision_store.tail(limit=10):
        try:
            recent_actions.append(parse_decision_type(row.action))
        except HTTPException:
            # Ignore unknown / legacy safely
            continue

    decision = decision_engine.decide(
        batch_index=0,
        trust_score=0.92,
        f1=0.84,
        f1_baseline=0.86,
        drift_score=0.02,
        recent_actions=recent_actions,
    )

    return {
        "mode": "simulation",
        "action": decision.action,
        "reason": decision.reason,
        "would_execute": decision_to_model_action(decision.action)
        != ModelAction.NONE,
    }


# ---------------------------------------------------------------------
# Phase IV.2 — Explicit manual actions (SAFE)
# ---------------------------------------------------------------------

@router.post("/models/promote")
def promote_model():
    version = model_store.promote_candidate({})
    return {"promoted_version": version}


@router.post("/models/rollback")
def rollback_model(version: str):
    model_store.rollback(version)
    return {"rolled_back_to": version}
