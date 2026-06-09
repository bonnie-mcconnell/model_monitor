"""FastAPI dashboard routes: metrics, decisions, model actions."""

from __future__ import annotations

import os
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from model_monitor.config.settings import load_config
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.core.decisions import DecisionType
from model_monitor.core.model_actions import ModelAction
from model_monitor.monitoring.thresholds import (
    CRITICAL_TRUST_SCORE,
    MAX_OUTPUT_DRIFT_SCORE,
    MIN_CONFORMAL_COVERAGE,
    MIN_DATA_QUALITY_SCORE,
    MIN_TRUST_SCORE,
)
from model_monitor.monitoring.trust_score import compute_trust_score
from model_monitor.storage.alert_store import AlertStore
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_store import MetricsStore
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)
from model_monitor.storage.metrics_summary_store import MetricsSummaryStore
from model_monitor.storage.model_store import ModelStore

_DASHBOARD_KEY_ENV = "MONITOR_DASHBOARD_KEY"


def _require_dashboard_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Optional dashboard authentication.

    When ``MONITOR_DASHBOARD_KEY`` is set, all ``/dashboard/*`` routes require
    a matching ``X-Api-Key`` header.  When the variable is unset the dashboard
    is unauthenticated - suitable for local development and internal-only
    deployments protected at the network layer.

    This mirrors the ingest endpoint's pattern so operators can use the same
    key rotation mechanism for both surfaces.

    Raises:
        HTTPException 401: if the key is set and the provided value does not match.
    """
    expected = os.environ.get(_DASHBOARD_KEY_ENV)
    if not expected:
        return  # auth disabled - no key configured
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing dashboard API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# Apply auth to the entire router so every endpoint inherits it without
# requiring per-route Depends() annotations.
router = APIRouter(
    prefix="/dashboard",
    tags=["dashboard"],
    dependencies=[Depends(_require_dashboard_key)],
)


def parse_decision_type(value: str) -> DecisionType:
    """Parse a raw string into a DecisionType, raising HTTP 400 on invalid values.

    Uses the enum's own membership test rather than a hand-maintained set,
    so adding a new DecisionType member automatically extends the valid set here.
    """
    try:
        return DecisionType(value)
    except ValueError:
        valid = [d.value for d in DecisionType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision action '{value}'. Valid actions: {valid}",
        )


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
    return {col.key: getattr(obj, col.key) for col in obj.__table__.columns}


# ---------------------------------------------------------------------
# Stores & services - lazy singletons
#
# Constructed on first request, not at module import time.  Import-time
# construction creates SQLite files, models/ directories, and config reads
# as a side-effect of importing the module - which breaks test isolation
# and causes spurious filesystem mutations in any environment that imports
# the FastAPI app without intending to run it.
# ---------------------------------------------------------------------

_metrics_store: MetricsStore | None = None
_summary_store: MetricsSummaryStore | None = None
_history_store: MetricsSummaryHistoryStore | None = None
_decision_store: DecisionStore | None = None
_model_store: ModelStore | None = None
_decision_engine: DecisionEngine | None = None


def _get_metrics_store() -> MetricsStore:
    global _metrics_store
    if _metrics_store is None:
        _metrics_store = MetricsStore()
    return _metrics_store


def _get_summary_store() -> MetricsSummaryStore:
    global _summary_store
    if _summary_store is None:
        _summary_store = MetricsSummaryStore()
    return _summary_store


def _get_history_store() -> MetricsSummaryHistoryStore:
    global _history_store
    if _history_store is None:
        _history_store = MetricsSummaryHistoryStore()
    return _history_store


def _get_decision_store() -> DecisionStore:
    global _decision_store
    if _decision_store is None:
        _decision_store = DecisionStore()
    return _decision_store


def _get_model_store() -> ModelStore:
    global _model_store
    if _model_store is None:
        _model_store = ModelStore()
    return _model_store


_alert_store: AlertStore | None = None


def _get_alert_store() -> AlertStore:
    global _alert_store
    if _alert_store is None:
        _alert_store = AlertStore()
    return _alert_store


def _get_decision_engine() -> DecisionEngine:
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = DecisionEngine(config=load_config())
    return _decision_engine


# ---------------------------------------------------------------------
# Metrics (READ-ONLY)
# ---------------------------------------------------------------------


@router.get("/metrics/tail")
def get_metrics_tail(
    limit: int = Query(100, ge=1, le=1000),
) -> list[Any]:
    return _get_metrics_store().tail(limit=limit)


@router.get("/metrics/latest")
def get_latest_metric() -> Any:
    return _get_metrics_store().latest()


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

    records, next_cursor = _get_metrics_store().list(
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
            {"timestamp": next_cursor[0], "id": next_cursor[1]} if next_cursor else None
        ),
    }


# ---------------------------------------------------------------------
# Aggregated metrics
# ---------------------------------------------------------------------


@router.get("/metrics/summary/{window}")
def get_metrics_summary(window: str) -> dict[str, Any]:
    summary = _get_summary_store().get(window)
    if summary is None:
        raise HTTPException(404, f"No summary for window '{window}'")
    return _orm_to_dict(summary)


@router.get("/metrics/summary/{window}/history")
def get_metrics_summary_history(
    window: str,
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    rows = _get_history_store().list_history(window=window, limit=limit)
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
    rows = _get_decision_store().tail(limit)
    return [_orm_to_dict(r) for r in reversed(rows)]


# ---------------------------------------------------------------------
# Alert history
# ---------------------------------------------------------------------


@router.get("/alerts/history")
def get_alert_history(
    limit: int = Query(100, ge=1, le=1000),
    severity: str | None = Query(None),
    window: str | None = Query(None),
    since_ts: float | None = Query(None),
) -> dict[str, Any]:
    """
    Return recent fired alerts, newest first.

    Query parameters:
    - severity: filter to "warning" or "critical"
    - window:   filter to a specific aggregation window (e.g. "5m")
    - since_ts: Unix timestamp; only return alerts after this time
    """
    alerts = _get_alert_store().tail(
        limit=limit,
        severity=severity,
        window=window,
        since_ts=since_ts,
    )
    return {"items": alerts, "count": len(alerts)}


# ---------------------------------------------------------------------
# Decision simulation
# ---------------------------------------------------------------------


@router.post("/decisions/simulate")
def simulate_decision() -> dict[str, Any]:
    """Simulate what decision the engine would make right now.

    Reads the most recent metric record and current baseline F1 from
    active.json.  Uses the actual decision count from the store as
    batch_index so cooldown logic behaves identically to the live
    aggregation loop - using batch_index=0 would always trigger the
    ephemeral cooldown if a retrain had previously fired.

    The response includes ``trust_components`` - the full per-signal
    breakdown - so the Streamlit dashboard can show exactly which signal
    is driving the decision, not just the scalar score.
    """
    ms = _get_metrics_store()
    ds = _get_decision_store()

    latest = ms.latest()
    if latest is None:
        return {
            "mode": "simulation",
            "action": "none",
            "reason": "no_metrics_available",
            "would_execute": False,
        }

    active_meta = _get_model_store().get_active_metadata()
    baseline_f1: float = (
        active_meta.get("metrics", {}).get("baseline_f1") or latest["f1"]
    )

    recent_actions: list[DecisionType] = []
    for row in ds.tail(limit=10):
        try:
            recent_actions.append(parse_decision_type(row.action))
        except HTTPException:
            continue

    # Use the real decision count as batch_index so cooldown arithmetic
    # matches the live system (batch_index=0 would make since_last negative
    # and always trigger the ephemeral cooldown after any prior retrain).
    batch_index = ds.count()

    # Pull behavioral violation rate from the most recent record.
    # Defaults to 0.0 when absent so the endpoint is backward-compatible
    # with records written before the BM branch existed.
    behavioral_violation_rate: float = float(
        latest.get("behavioral_violation_rate") or 0.0
    )

    trust_score, trust_components = compute_trust_score(
        accuracy=latest.get("accuracy", latest["f1"]),
        f1=latest["f1"],
        avg_confidence=latest.get("avg_confidence", 1.0),
        drift_score=latest["drift_score"],
        decision_latency_ms=latest.get("decision_latency_ms", 0.0),
        behavioral_violation_rate=behavioral_violation_rate,
        config=_get_decision_engine().cfg.trust_score,
    )

    candidate_exists = _get_model_store().has_candidate()

    decision = _get_decision_engine().decide(
        batch_index=batch_index,
        trust_score=trust_score,
        f1=latest["f1"],
        f1_baseline=baseline_f1,
        drift_score=latest["drift_score"],
        recent_actions=recent_actions,
        candidate_exists=candidate_exists,
    )

    inputs: dict[str, Any] = {
        "f1": latest["f1"],
        "f1_baseline": baseline_f1,
        "drift_score": latest["drift_score"],
        "batch_index": batch_index,
    }
    # Include behavioral_violation_rate in inputs only when the field is
    # present in the latest record so the response is backward-compatible
    # with main-branch clients that never write this field.
    if latest.get("behavioral_violation_rate") is not None:
        inputs["behavioral_violation_rate"] = behavioral_violation_rate

    return {
        "mode": "simulation",
        "action": decision.action,
        "reason": decision.reason,
        "trust_score": trust_score,
        "trust_components": dict(trust_components),
        "candidate_exists": candidate_exists,
        "inputs": inputs,
        "would_execute": decision_to_model_action(decision.action) != ModelAction.NONE,
    }


# ---------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------


@router.get("/models/active")
def get_active_model() -> dict[str, Any]:
    """Return the currently active model version and its metadata."""
    meta = _get_model_store().get_active_metadata()
    if not meta:
        raise HTTPException(404, "No active model found")
    return meta


@router.get("/config")
def get_config() -> dict[str, Any]:
    """Return the live operational configuration.

    Exposes all policy thresholds so operators can understand exactly what
    values govern current retrain, rollback, and drift decisions without
    reading YAML files.  The dashboard shows trust scores and action history;
    this endpoint shows *why* those actions fired.

    The ``retrain.evidence_window`` and ``retrain.min_samples`` fields are the
    two parameters most commonly adjusted in production:
    - Reduce ``evidence_window`` to retrain faster after detecting degradation.
    - Increase ``min_samples`` if you want more labeled data before retraining.
    """
    cfg = _get_decision_engine().cfg
    return {
        "drift": {
            "psi_threshold": cfg.drift.psi_threshold,
            "window_batches": cfg.drift.window,
            "description": "PSI >= psi_threshold triggers reject action",
        },
        "retrain": {
            "min_f1_gain": cfg.retrain.min_f1_gain,
            "cooldown_batches": cfg.retrain.cooldown_batches,
            "evidence_window": cfg.retrain.evidence_window,
            "min_samples": cfg.retrain.min_samples,
            "min_stable_batches": cfg.retrain.min_stable_batches,
            "description": (
                "Retrain fires when evidence_window consecutive summaries show "
                "degradation, and min_samples labeled rows are available. "
                "Dual cooldown: batch-index (ephemeral) + recent_actions (durable)."
            ),
        },
        "rollback": {
            "max_f1_drop": cfg.rollback.max_f1_drop,
            "description": "F1 drop >= max_f1_drop triggers immediate rollback",
        },
        "trust_score": {
            "weights": {
                "accuracy": cfg.trust_score.accuracy,
                "f1": cfg.trust_score.f1,
                "calibration": cfg.trust_score.calibration,
                "drift": cfg.trust_score.drift,
                "latency": cfg.trust_score.latency,
                "data_quality": cfg.trust_score.data_quality,
                "behavioral": cfg.trust_score.behavioral,
            },
            "drift_to_trust": {
                "stable_below": 0.1,
                "severe_above": 0.3,
                "description": "Linear decay between 0.1 and 0.3",
            },
            "latency_to_trust": {
                "full_score_below_ms": 300,
                "zero_score_above_ms": 1500,
                "description": "Linear decay between 300ms and 1500ms",
            },
        },
        "alerting": {
            "min_trust_score": MIN_TRUST_SCORE,
            "critical_trust_score": CRITICAL_TRUST_SCORE,
            "min_conformal_coverage": MIN_CONFORMAL_COVERAGE,
            "min_data_quality_score": MIN_DATA_QUALITY_SCORE,
            "max_output_drift_score": MAX_OUTPUT_DRIFT_SCORE,
            "cooldown_seconds": 300,
        },
    }


@router.get("/models/versions")
def list_model_versions() -> list[dict[str, str]]:
    """
    List all archived model versions, newest first.

    Each entry contains version (timestamp string), path, and created_at.
    The currently active version is available via GET /dashboard/models/active.
    """
    return _get_model_store().list_versions()


@router.get("/health/detailed")
def get_detailed_health() -> dict[str, Any]:
    """
    Single-payload operational status for status pages and runbooks.

    Returns model state, current trust scores across all windows, live
    thresholds, and alert rate - everything an on-call engineer needs to
    assess system health in one request.
    """
    ms = _get_metrics_store()
    ds = _get_decision_store()
    model_store = _get_model_store()
    alert_store = _get_alert_store()
    summary_store = _get_summary_store()

    # Model state
    active_meta = model_store.get_active_metadata()
    model_ready = bool(active_meta)

    # Trust scores per window
    windows: dict[str, Any] = {}
    for window in ("5m", "1h", "24h"):
        summary = summary_store.get(window)
        if summary and summary.n_batches > 0:
            windows[window] = {
                "trust_score": summary.trust_score,
                "avg_f1": summary.avg_f1,
                "avg_drift_score": summary.avg_drift_score,
                "n_batches": summary.n_batches,
            }
        else:
            windows[window] = None

    # Alert rates
    now = time.time()
    alerts_1h = alert_store.count_since(now - 3600)
    alerts_24h = alert_store.count_since(now - 86400)
    critical_24h = alert_store.count_since(now - 86400, severity="critical")

    # Latest batch
    latest = ms.latest()

    return {
        "model": {
            "ready": model_ready,
            "version": active_meta.get("version") if active_meta else None,
            "baseline_f1": active_meta.get("metrics", {}).get("baseline_f1")
            if active_meta
            else None,
            "promoted_at": active_meta.get("promoted_at_utc") if active_meta else None,
        },
        "trust_scores": windows,
        "latest_batch": {
            "batch_id": latest["batch_id"] if latest else None,
            "timestamp": latest["timestamp"] if latest else None,
            "drift_score": latest["drift_score"] if latest else None,
            "action": latest["action"] if latest else None,
        }
        if latest
        else None,
        "alerts": {
            "last_1h": alerts_1h,
            "last_24h": alerts_24h,
            "critical_last_24h": critical_24h,
        },
        "total_decisions": ds.count(),
    }


@router.get("/models/compare")
def compare_models(v1: str, v2: str) -> dict[str, Any]:
    """
    Compare two archived model versions.

    Returns the metadata stored at promotion time for each version so
    operators can answer "was v2 actually better than v1 when it was
    promoted?" without reconstructing state from raw metrics tables.

    Both versions must exist in the archive.  The active (current) version
    can be identified via GET /dashboard/models/active and compared here.
    """
    store = _get_model_store()
    versions = {v["version"]: v for v in store.list_versions()}

    missing = [v for v in (v1, v2) if v not in versions]
    if missing:
        raise HTTPException(404, f"Version(s) not found in archive: {missing}")

    meta_active = store.get_active_metadata()
    active_version = meta_active.get("version")

    def _enrich(version: str) -> dict[str, Any]:
        info: dict[str, Any] = dict(versions[version])
        info["is_active"] = bool(version == active_version)
        return info

    result: dict[str, Any] = {
        "v1": _enrich(v1),
        "v2": _enrich(v2),
    }

    # Surface the baseline_f1 delta if both versions have it stored
    f1_v1 = None
    f1_v2 = None
    # active.json stores baseline_f1 for the current model; archived models
    # don't carry this - we surface what we have
    if v1 == active_version:
        f1_v1 = meta_active.get("metrics", {}).get("baseline_f1")
    if v2 == active_version:
        f1_v2 = meta_active.get("metrics", {}).get("baseline_f1")

    if f1_v1 is not None and f1_v2 is not None:
        result["f1_delta"] = round(f1_v2 - f1_v1, 6)
        result["f1_delta_direction"] = (
            "improved"
            if f1_v2 > f1_v1
            else "regressed"
            if f1_v2 < f1_v1
            else "unchanged"
        )

    return result


@router.post("/models/promote")
def promote_model() -> dict[str, str]:
    version = _get_model_store().promote_candidate({})
    return {"promoted_version": version}


@router.post("/models/rollback")
def rollback_model(version: str) -> dict[str, str]:
    _get_model_store().rollback(version)
    return {"rolled_back_to": version}


# ---------------------------------------------------------------------
# Shadow mode
# ---------------------------------------------------------------------

_shadow_predictor: Any | None = None


def _get_shadow_predictor() -> Any:
    """
    Return the module-level ShadowPredictor singleton.

    Instantiated lazily on first request so that importing the dashboard
    module does not construct a Predictor (which loads the model from disk).

    Accumulates shadow statistics across requests - callers read
    ``shadow_stats`` to see running aggregates.  Stats are reset by
    POST /dashboard/shadow/reset.
    """
    global _shadow_predictor
    if _shadow_predictor is None:
        from model_monitor.config.settings import load_config as _lc
        from model_monitor.inference.predict import Predictor as _P
        from model_monitor.inference.shadow import ShadowPredictor as _SP

        cfg = _lc()
        primary = _P(config=cfg)
        primary.reload()
        _shadow_predictor = _SP(primary=primary)
    return _shadow_predictor


@router.get("/causal-drift/latest")
async def causal_drift_latest() -> dict[str, object]:
    """Return the most recent causal drift attribution report.

    When the Predictor is configured with a ``CausalDriftAttributor`` and drift
    has been detected, this endpoint returns the attribution - classifying each
    drifting feature as ``genuine_shift``, ``pipeline_suspect``, or
    ``correlated_follower``.

    Returns ``{"available": false}`` when no attribution has been computed yet
    (e.g., no drift detected, or attributor not configured).
    """
    latest = _get_metrics_store().tail(limit=1)
    if not latest:
        return {"available": False, "reason": "no_batches_recorded"}

    rec = latest[0]
    causal = rec.get("causal_drift_report")
    if causal is None:
        return {
            "available": False,
            "reason": "causal_drift_attributor_not_configured_or_no_drift",
        }
    return {"available": True, "report": causal}


@router.get("/threshold-advisor/status")
async def threshold_advisor_status() -> dict[str, object]:
    """Return the current status of the adaptive threshold advisor.

    The ThresholdAdvisor accumulates stable-period observations and, once it
    has enough data (default: 30 batches), produces data-driven threshold
    recommendations.  This endpoint exposes:
      - How many stable-period observations have been recorded.
      - Whether enough observations exist for a recommendation.
      - The recommendation itself (when available).

    Operators can use this to decide when to update their ``drift.yaml`` and
    ``trust_score.yaml`` threshold values with calibrated alternatives.
    """
    # ThresholdAdvisor is an optional attribute on the Predictor singleton.
    # Resolve it dynamically so this endpoint degrades cleanly when not configured.
    advisor = None
    try:
        import importlib  # noqa: PLC0415

        _startup = importlib.import_module("model_monitor.api.startup")
        _pred = getattr(_startup, "_predictor", None)
        if _pred is not None:
            advisor = getattr(_pred, "_threshold_advisor", None)
    except Exception:
        pass

    if advisor is None:
        return {"available": False, "reason": "threshold_advisor_not_configured"}

    status: dict[str, object] = {
        "available": True,
        "n_observations": advisor.n_observations,
        "is_ready": advisor.is_ready,
        "min_batches_required": advisor.min_batches,
        "alpha": advisor.alpha,
        "feature_names": advisor.feature_names,
    }

    if advisor.is_ready:
        try:
            rec = advisor.recommend()
            status["recommendation"] = {
                "psi_warn_global": rec.psi_warn_global,
                "psi_warn_per_feature": dict(
                    zip(rec.feature_names, rec.psi_warn_per_feature)
                ),
                "trust_warn": rec.trust_warn,
                "trust_critical": rec.trust_critical,
                "notes": list(rec.notes),
            }
        except Exception as exc:
            status["recommendation_error"] = str(exc)

    return status


@router.get("/shadow")
def get_shadow_stats() -> dict[str, Any]:
    """
    Return aggregated shadow-mode comparison statistics.

    Shadow mode runs a candidate model silently alongside the primary on
    every live batch.  This endpoint exposes running aggregates so the
    Streamlit dashboard can judge whether the candidate is ready for
    promotion without interrupting live traffic.

    The endpoint never returns 404 - when no candidate is loaded it returns
    ``has_candidate: false`` and zeroed stats so dashboards can poll
    unconditionally without error-handling branches.

    Response fields
    ---------------
    has_candidate         : bool  - whether a candidate model is loaded
    n_batches             : int   - shadow-evaluated batch count
    total_samples         : int   - total data points observed
    mean_agreement_rate   : float - fraction of predictions that agree
    mean_primary_f1       : float - running mean F1 for the primary
    mean_candidate_f1     : float - running mean F1 for the candidate
    mean_primary_trust    : float - running mean trust score, primary
    mean_candidate_trust  : float - running mean trust score, candidate
    candidate_beats_primary: bool - True when candidate wins on F1 and trust
    recommendation        : str  - promote_candidate | keep_primary | no_data
    """
    sp = _get_shadow_predictor()
    stats = sp.shadow_stats

    if stats.n_batches == 0:
        recommendation = "no_data"
    elif not sp.has_candidate():
        recommendation = "no_candidate_loaded"
    elif stats.candidate_beats_primary:
        recommendation = "promote_candidate"
    else:
        recommendation = "keep_primary"

    return {
        "has_candidate": sp.has_candidate(),
        "n_batches": stats.n_batches,
        "total_samples": stats.total_samples,
        "mean_agreement_rate": round(stats.mean_agreement_rate, 4),
        "mean_primary_f1": round(stats.mean_primary_f1, 4),
        "mean_candidate_f1": round(stats.mean_candidate_f1, 4),
        "mean_primary_trust": round(stats.mean_primary_trust, 4),
        "mean_candidate_trust": round(stats.mean_candidate_trust, 4),
        "candidate_beats_primary": stats.candidate_beats_primary,
        "recommendation": recommendation,
    }


@router.post("/shadow/reset")
def reset_shadow_stats() -> dict[str, str]:
    """
    Reset accumulated shadow statistics.

    Call after a promotion decision to start a fresh comparison window for
    the newly promoted model against any future candidate.
    """
    _get_shadow_predictor().reset_shadow_stats()
    return {"status": "reset"}


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral contract summary (behavior-monitoring branch only)
# ─────────────────────────────────────────────────────────────────────────────


_WINDOW_SECONDS: dict[str, int] = {
    "5m": 300,
    "1h": 3600,
    "24h": 86_400,
}


# ─────────────────────────────────────────────────────────────────────────────
# Model card endpoints
# ─────────────────────────────────────────────────────────────────────────────


def _model_card_path(version: int) -> str:
    """Return the filesystem path for a model card JSON file."""
    base = os.environ.get("MODEL_STORE_DIR", "data/models")
    return os.path.join(base, f"v{version}_card.json")


@router.get("/models/{version}/card")
async def get_model_card(version: int) -> dict[str, object]:
    """Return the model card for a specific model version.

    The card captures training provenance: dataset hash, feature schema,
    evaluation metrics at promotion, and the reason this version was promoted.

    Returns 404 when no card exists for the requested version (models trained
    before cards were introduced, or version does not exist).
    """
    from fastapi import HTTPException

    from model_monitor.training.model_card import ModelCard

    path = _model_card_path(version)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"No model card found for version {version}. "
            f"Cards are written at promotion time from v8 onward.",
        )
    try:
        card = ModelCard.load(path)
        return {"available": True, "card": card.summary_dict()}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model card: {exc}",
        ) from exc


@router.get("/models/cards/all")
async def list_model_cards() -> dict[str, object]:
    """Return summary dicts for all available model cards, sorted by version.

    Models trained before v8 will not have cards.
    """
    from model_monitor.training.model_card import ModelCard

    base = os.environ.get("MODEL_STORE_DIR", "data/models")
    cards = []
    if os.path.isdir(base):
        for entry in sorted(os.listdir(base)):
            if entry.endswith("_card.json"):
                try:
                    card = ModelCard.load(os.path.join(base, entry))
                    cards.append(card.summary_dict())
                except Exception:
                    pass
    return {"n_cards": len(cards), "cards": cards}


# ─────────────────────────────────────────────────────────────────────────────
# Regression monitoring endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/regression/latest")
async def regression_latest() -> dict[str, object]:
    """Return the most recent batch result from the regression monitor.

    Returns ``available: False`` when no regression monitor is configured.
    The regression monitor is wired in by passing ``regression_monitor`` to
    the ``Predictor`` constructor (or by using ``RegressionMonitor`` directly
    with ``db_path`` set to the same database).
    """
    import model_monitor.api.startup as _startup

    pred = getattr(_startup, "_predictor", None)
    reg_monitor = (
        getattr(pred, "_regression_monitor", None) if pred is not None else None
    )

    if reg_monitor is None:
        return {"available": False, "reason": "regression_monitor_not_configured"}

    history = reg_monitor.history
    if not history:
        return {"available": False, "reason": "no_batches_recorded"}

    latest = history[0]
    return {
        "available": True,
        "batch_id": latest.get("batch_id"),
        "n_samples": latest.get("n_samples"),
        "mae": latest.get("mae"),
        "rmse": latest.get("rmse"),
        "wasserstein": latest.get("wasserstein"),
        "trust_score": latest.get("trust_score"),
        "coverage_rate": latest.get("coverage_rate"),
        "n_batches": reg_monitor.n_batches,
    }


@router.get("/regression/summary")
async def regression_summary() -> dict[str, object]:
    """Return rolling summary statistics from the regression monitor."""
    import model_monitor.api.startup as _startup

    pred = getattr(_startup, "_predictor", None)
    reg_monitor = (
        getattr(pred, "_regression_monitor", None) if pred is not None else None
    )

    if reg_monitor is None:
        return {"available": False, "reason": "regression_monitor_not_configured"}

    return {"available": True, **reg_monitor.summary()}


# ─────────────────────────────────────────────────────────────────────────────
# Population drift trend endpoint
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/drift/population")
async def drift_population(
    limit: int = 30,
) -> dict[str, object]:
    """Return per-feature PSI trend over the last ``limit`` batches.

    This is the primary endpoint for drift *investigation* (as opposed to
    detection).  When PSI spikes on a single feature, this endpoint tells
    you whether the shift was sudden (one-batch spike) or gradual
    (slowly increasing PSI over many batches).

    Returns a dict with:
      - ``feature_names``:  list of feature names.
      - ``batches``:        list of batch IDs in chronological order.
      - ``psi_by_feature``: dict mapping feature name → list of PSI values
                            (one per batch, aligned with ``batches``).
                            ``null`` entries mean the drift window was not yet
                            full for that batch.
      - ``max_psi_per_feature``: peak PSI per feature over the window.
      - ``mean_psi_per_feature``: mean PSI per feature over the window.
      - ``n_batches``: number of batches returned.
    """
    store = _get_metrics_store()
    records = store.tail(limit=limit)
    if not records:
        return {
            "n_batches": 0,
            "feature_names": [],
            "batches": [],
            "psi_by_feature": {},
            "max_psi_per_feature": {},
            "mean_psi_per_feature": {},
        }

    # Collect feature names from the first record that has per-feature scores.
    feature_names: list[str] = []
    for r in records:
        fscores = r.get("feature_drift_scores")
        if isinstance(fscores, list) and fscores:
            feature_names = [f"f{i}" for i in range(len(fscores))]
            break

    batches = [str(r.get("batch_id", "")) for r in records]
    psi_by_feature: dict[str, list[float | None]] = {name: [] for name in feature_names}

    for r in records:
        fscores = r.get("feature_drift_scores")
        if isinstance(fscores, list) and len(fscores) == len(feature_names):
            for name, val in zip(feature_names, fscores):
                psi_by_feature[name].append(float(val))
        else:
            for name in feature_names:
                psi_by_feature[name].append(None)

    # Summary statistics (ignoring null entries)
    max_psi: dict[str, float] = {}
    mean_psi: dict[str, float] = {}
    for name, vals in psi_by_feature.items():
        real = [v for v in vals if v is not None]
        max_psi[name] = max(real) if real else 0.0
        mean_psi[name] = sum(real) / len(real) if real else 0.0

    return {
        "n_batches": len(records),
        "feature_names": feature_names,
        "batches": batches,
        "psi_by_feature": psi_by_feature,
        "max_psi_per_feature": max_psi,
        "mean_psi_per_feature": mean_psi,
    }
