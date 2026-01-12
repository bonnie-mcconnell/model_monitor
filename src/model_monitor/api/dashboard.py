from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from fastapi import APIRouter, Query, HTTPException

from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.monitoring.types import MetricRecord
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.model_action_executor import ModelActionExecutor
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
# Helpers
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
    return value  # type: ignore[return-value]


def decision_to_model_action(action: DecisionType) -> ModelAction:
    mapping = {
        "none": ModelAction.NONE,
        "retrain": ModelAction.RETRAIN,
        "promote": ModelAction.PROMOTE,
        "rollback": ModelAction.ROLLBACK,
        "reject": ModelAction.REJECT,
    }
    return mapping.get(action, ModelAction.NONE)


# ---------------------------------------------------------------------
# Stores & services (singleton per process)
# ---------------------------------------------------------------------

metrics_store = MetricsStore()
summary_store = MetricsSummaryStore()
decision_store = DecisionStore()
model_store = ModelStore()
retrain_pipeline = RetrainPipeline()

app_config = load_config()
decision_engine = DecisionEngine(config=app_config)

# ---------------------------------------------------------------------
# Metrics ingestion
# ---------------------------------------------------------------------

@router.post("/metrics")
def write_metric(
    batch_id: str,
    n_samples: int,
    accuracy: float,
    f1: float,
    avg_confidence: float,
    drift_score: float,
    decision_latency_ms: float,
    action: str = Query("none"),
    reason: str = "",
):
    decision_action = parse_decision_type(action)

    record: MetricRecord = {
        "timestamp": datetime.utcnow().timestamp(),
        "batch_id": batch_id,
        "n_samples": n_samples,
        "accuracy": accuracy,
        "f1": f1,
        "avg_confidence": avg_confidence,
        "drift_score": drift_score,
        "decision_latency_ms": decision_latency_ms,
        "action": decision_action,
        "reason": reason,
        "previous_model": None,
        "new_model": None,
    }

    metrics_store.write(record)

    decision_store.record(
        decision=Decision(
            action=decision_action,
            reason=reason,
            metadata={},
        ),
        model_version=model_store.get_active_version(),
    )

    return {"status": "ok"}

# ---------------------------------------------------------------------
# Metrics access
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
    cursor = None
    if cursor_ts is not None and cursor_id is not None:
        cursor = (cursor_ts, cursor_id)

    typed_action: Optional[DecisionType] = (
        parse_decision_type(action) if action is not None else None
    )

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
# Decisions
# ---------------------------------------------------------------------

@router.get("/decisions/history")
def get_decision_history(limit: int = Query(100, ge=1, le=1000)):
    rows = decision_store.tail(limit)
    return [r.__dict__ for r in reversed(rows)]

# ---------------------------------------------------------------------
# Phase IV.1 — Decision simulation
# ---------------------------------------------------------------------

@router.post("/decisions/simulate")
def simulate_decision():
    recent_actions: list[DecisionType] = [
        parse_decision_type(r.action)
        for r in decision_store.tail(limit=10)
    ]


    decision = decision_engine.decide(
        batch_index=0,
        trust_score=0.92,
        f1=0.84,
        f1_baseline=0.86,
        drift_score=0.02,
        recent_actions=recent_actions,
    )

    action = decision_to_model_action(decision.action)

    executor = ModelActionExecutor(
        model_store=model_store,
        retrain_pipeline=retrain_pipeline,
        decision_store=decision_store,
        dry_run=True,
    )

    executor.execute(action=action, context={})

    return {
        "mode": "simulation",
        "action": decision.action,
        "reason": decision.reason,
        "executed": action != ModelAction.NONE,
        "dry_run": True,
    }

# ---------------------------------------------------------------------
# Phase IV.2 — REAL execution
# ---------------------------------------------------------------------

@router.post("/decisions/execute")
def execute_decision():
    rows = decision_store.tail(limit=1)
    if not rows:
        raise HTTPException(404, "No decision to execute")

    decision = rows[0]
    decision_action = parse_decision_type(decision.action)
    action = decision_to_model_action(decision_action)

    executor = ModelActionExecutor(
        model_store=model_store,
        retrain_pipeline=retrain_pipeline,
        decision_store=decision_store,
        dry_run=False,
    )

    result = executor.execute(action=action, context={})

    return {
        "executed": True,
        "action": decision.action,
        "result": result,
    }

# ---------------------------------------------------------------------
# Phase IV.3 — Explicit promotion / rollback
# ---------------------------------------------------------------------

@router.post("/models/promote")
def promote_model():
    version = model_store.promote_candidate({})
    return {"promoted_version": version}


@router.post("/models/rollback")
def rollback_model(version: str):
    model_store.rollback(version)
    return {"rolled_back_to": version}
