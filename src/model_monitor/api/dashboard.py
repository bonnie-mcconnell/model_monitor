from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, Sequence

from fastapi import APIRouter, Query, HTTPException

from model_monitor.monitoring.types import MetricRecord
from model_monitor.core.decisions import Decision, DecisionType
from model_monitor.core.model_action_executor import ModelActionExecutor
from model_monitor.core.decision_engine import DecisionEngine

from model_monitor.config.settings import load_config

from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.model_store import ModelStore
from model_monitor.training.retrain_pipeline import RetrainPipeline
from model_monitor.storage.models.metrics_summary_history import (
    MetricsSummaryHistoryORM,
)
from model_monitor.core.model_actions import ModelAction

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# ---------------------------------------------------------------------
# Stores & services (singleton per process)
# ---------------------------------------------------------------------

metrics_store = MetricsStore()
summary_store = MetricsSummaryStore()
decision_store = DecisionStore()
model_store = ModelStore()
retrain_pipeline = RetrainPipeline()

# Load app config once
app_config = load_config()

# Decision policy engine (pure logic)
decision_engine = DecisionEngine(config=app_config)

# ---------------------------------------------------------------------
# Metrics ingestion (internal / admin)
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
    action: DecisionType = "none",
    reason: str = "",
):
    record: MetricRecord = {
        "timestamp": datetime.utcnow().timestamp(),
        "batch_id": batch_id,
        "n_samples": n_samples,
        "accuracy": accuracy,
        "f1": f1,
        "avg_confidence": avg_confidence,
        "drift_score": drift_score,
        "decision_latency_ms": decision_latency_ms,
        "action": action,
        "reason": reason,
        "previous_model": None,
        "new_model": None,
    }

    metrics_store.write(record)

    decision_store.record(
        decision=Decision(
            action=action,
            reason=reason,
            metadata={},
        ),
        model_version=model_store.get_active_version(),
    )

    return {"status": "ok"}

# ---------------------------------------------------------------------
# Raw metrics access
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
    action: Optional[DecisionType] = Query(None),
    model: Optional[str] = Query(None),
    min_accuracy: Optional[float] = Query(None, ge=0.0, le=1.0),
    start_ts: Optional[float] = Query(None),
    end_ts: Optional[float] = Query(None),
):
    cursor = None
    if cursor_ts is not None and cursor_id is not None:
        cursor = (cursor_ts, cursor_id)

    records, next_cursor = metrics_store.list(
        limit=limit,
        cursor=cursor,
        action=action,
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
        raise HTTPException(
            status_code=404,
            detail=f"No summary available for window '{window}'",
        )
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
        "items": [
            {
                "window": r.window,
                "timestamp": r.timestamp,
                "n_batches": r.n_batches,
                "avg_accuracy": r.avg_accuracy,
                "avg_f1": r.avg_f1,
                "avg_confidence": r.avg_confidence,
                "avg_drift_score": r.avg_drift_score,
                "avg_latency_ms": r.avg_latency_ms,
            }
            for r in reversed(rows)
        ],
    }

# ---------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------

@router.get("/decisions/summary")
def get_decision_summary() -> Dict[str, int]:
    rows = decision_store.tail(limit=10_000)
    summary: Dict[str, int] = {}
    for r in rows:
        summary[r.action] = summary.get(r.action, 0) + 1
    return summary


@router.get("/decisions/history")
def get_decision_history(limit: int = Query(100, ge=1, le=1000)):
    rows = decision_store.tail(limit)
    return [
        {
            "timestamp": r.timestamp,
            "batch_index": r.batch_index,
            "action": r.action,
            "reason": r.reason,
            "trust_score": r.trust_score,
            "f1": r.f1,
            "drift_score": r.drift_score,
            "model_version": r.model_version,
        }
        for r in reversed(rows)
    ]

# ---------------------------------------------------------------------
# Decision simulation (NO side effects)
# ---------------------------------------------------------------------

@router.post("/decisions/simulate")
def simulate_decision(payload: Dict[str, Any]):
    """
    Run full decision + execution pipeline in dry-run mode.
    No side effects.
    """

    decision: Decision = decision_engine.decide(
        batch_index=payload["batch_index"],
        trust_score=payload["trust_score"],
        f1=payload["f1"],
        f1_baseline=payload["f1_baseline"],
        drift_score=payload["drift_score"],
        recent_actions=payload.get("recent_actions"),
    )

    executor = ModelActionExecutor(
        model_store=model_store,
        retrain_pipeline=retrain_pipeline,
        decision_store=decision_store,
        dry_run=True,
    )

    executor.execute(
        action=ModelAction.from_decision(decision.action),
        context=payload,
    )

    return {
        "mode": "simulation",
        "action": decision.action,
        "reason": decision.reason,
    }
