"""model-monitor replay - offline decision replay from stored metrics.

Reads ``MetricsSummaryStore`` (the same rolling-window aggregates the live
server writes) and reruns ``DecisionEngine`` on each record within a time
range.  Output is a human-readable table to stdout; every decision is also
written to ``DecisionStore`` so the replay is captured in the audit log.

Usage
-----
::

    model-monitor replay
    model-monitor replay --from 2025-01-01
    model-monitor replay --from 2025-01-01T09:00 --to 2025-01-01T18:00
    model-monitor replay --window 1h --dry-run
    model-monitor replay --from 2025-01-01 --dry-run --json

Options
-------
--from ISO     Start timestamp (inclusive).  Defaults to 24 hours ago.
--to   ISO     End timestamp (inclusive).    Defaults to now.
--window STR   Aggregation window to replay: 5m, 1h, 24h.  Default: 1h.
--dry-run      Print decisions without writing to DecisionStore.
--json         Emit newline-delimited JSON instead of the table.

Exit codes
----------
0  Replay completed (even if no records matched the time range).
1  Configuration or storage error.
2  Invalid arguments.

Design notes
------------
The replay is intentionally read-only with respect to the model store - it
never calls ``ModelStore.promote_candidate()`` or triggers retrain pipelines.
Its purpose is to answer "what would the engine have decided?" for a past
time window, not to re-execute lifecycle actions.

The ``DecisionRunner`` already provides the correct ``run_once()`` API for
this.  This module wraps it with a date-range filter over
``MetricsSummaryHistoryStore`` (the persisted rolling-window history) so each
historical snapshot drives one engine call.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

from model_monitor.config.settings import load_config
from model_monitor.core.decision_engine import DecisionEngine
from model_monitor.storage.decision_store import DecisionStore
from model_monitor.storage.metrics_summary_history_store import (
    MetricsSummaryHistoryStore,
)
from model_monitor.storage.model_store import ModelStore

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_ACTION_COLOUR = {
    "none": "\033[32m",  # green
    "retrain": "\033[33m",  # yellow
    "promote": "\033[36m",  # cyan
    "rollback": "\033[31m",  # red
    "reject": "\033[35m",  # magenta
    "system_error": "\033[31m",  # red
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _colourise(action: str, text: str, *, no_colour: bool) -> str:
    if no_colour:
        return text
    colour = _ACTION_COLOUR.get(action, "")
    return f"{colour}{text}{_RESET}"


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_iso(value: str, *, field: str) -> float:
    """Parse an ISO 8601 date or datetime string to a UTC Unix timestamp."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    print(
        f"error: --{field} value {value!r} is not a recognised ISO date/datetime.\n"
        "       Use YYYY-MM-DD or YYYY-MM-DDTHH:MM[:SS].",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


_COL_WIDTHS = {
    "ts": 24,
    "window": 6,
    "trust": 7,
    "f1": 6,
    "in_psi": 7,
    "out_psi": 7,
    "dq": 6,
    "cp_cov": 7,
    "action": 14,
    "reason": 34,
}


def _header() -> str:
    return (
        f"{_BOLD}"
        f"{'Timestamp':<{_COL_WIDTHS['ts']}}"
        f"{'Win':>{_COL_WIDTHS['window']}}"
        f"{'Trust':>{_COL_WIDTHS['trust']}}"
        f"{'F1':>{_COL_WIDTHS['f1']}}"
        f"{'InPSI':>{_COL_WIDTHS['in_psi']}}"
        f"{'OutPSI':>{_COL_WIDTHS['out_psi']}}"
        f"{'DQ':>{_COL_WIDTHS['dq']}}"
        f"{'CvgR':>{_COL_WIDTHS['cp_cov']}}"
        f"  {'Action':<{_COL_WIDTHS['action']}}"
        f"  {'Reason'}"
        f"{_RESET}"
    )


def _divider() -> str:
    total = sum(_COL_WIDTHS.values()) + 4
    return "─" * total


def _row(
    ts: float,
    window: str,
    trust: float,
    f1: float,
    drift: float,
    action: str,
    reason: str,
    *,
    no_colour: bool,
    output_drift: float | None = None,
    data_quality: float | None = None,
    conformal_cov: float | None = None,
) -> str:
    action_col = _colourise(
        action, f"{action:<{_COL_WIDTHS['action']}}", no_colour=no_colour
    )
    _od = (
        f"{output_drift:>{_COL_WIDTHS['out_psi']}.4f}"
        if output_drift is not None
        else f"{'-':>{_COL_WIDTHS['out_psi']}}"
    )
    _dq = (
        f"{data_quality:>{_COL_WIDTHS['dq']}.4f}"
        if data_quality is not None
        else f"{'-':>{_COL_WIDTHS['dq']}}"
    )
    _cov = (
        f"{conformal_cov:>{_COL_WIDTHS['cp_cov']}.4f}"
        if conformal_cov is not None
        else f"{'-':>{_COL_WIDTHS['cp_cov']}}"
    )
    return (
        f"{_fmt_ts(ts):<{_COL_WIDTHS['ts']}}"
        f"{window:>{_COL_WIDTHS['window']}}"
        f"{trust:>{_COL_WIDTHS['trust']}.4f}"
        f"{f1:>{_COL_WIDTHS['f1']}.4f}"
        f"{drift:>{_COL_WIDTHS['in_psi']}.4f}"
        f"{_od}"
        f"{_dq}"
        f"{_cov}"
        f"  {action_col}"
        f"  {reason}"
    )


# ---------------------------------------------------------------------------
# Core replay logic
# ---------------------------------------------------------------------------


def _run_replay(
    *,
    from_ts: float,
    to_ts: float,
    window: str,
    dry_run: bool,
    emit_json: bool,
    no_colour: bool,
) -> int:
    """Execute the replay and write output to stdout.

    Returns:
        Exit code: 0 on success, 1 on storage/config error.
    """
    try:
        cfg = load_config()
        engine = DecisionEngine(config=cfg)
        history_store = MetricsSummaryHistoryStore()
        model_store = ModelStore()
        decision_store = DecisionStore()
    except Exception as exc:
        print(f"error: failed to initialise stores - {exc}", file=sys.stderr)
        return 1

    try:
        rows = history_store.query_range(
            window=window,
            from_ts=from_ts,
            to_ts=to_ts,
        )
    except Exception as exc:
        print(
            f"error: MetricsSummaryHistoryStore.query_range failed - {exc}",
            file=sys.stderr,
        )
        return 1

    if not rows:
        ts_from_str = _fmt_ts(from_ts)
        ts_to_str = _fmt_ts(to_ts)
        print(
            f"No {window!r} window records found between "
            f"{ts_from_str} and {ts_to_str}.",
            file=sys.stderr,
        )
        print(
            "\nTip: the history store is populated by the aggregation loop.\n"
            "     Make sure the server has been running ('make run') and\n"
            "     the simulation has completed at least one full window\n"
            "     ('make sim').  The 1h window closes after 60 minutes of\n"
            "     server uptime; the 5m window closes after 5 minutes.",
            file=sys.stderr,
        )
        return 0

    # Read active model metadata once - baseline F1 and candidate flag.
    active_meta = model_store.get_active_metadata()
    baseline_f1: float | None = active_meta.get("metrics", {}).get("baseline_f1")
    candidate_exists = model_store.has_candidate()

    # Seed recent_actions from the live decision store so hysteresis and
    # cooldowns reflect the real system state, not a blank slate.
    recent_rows = decision_store.tail(limit=10)
    from model_monitor.core.decisions import DecisionType

    recent_actions = [DecisionType(r.action) for r in recent_rows]

    replayed: list[dict[str, Any]] = []

    if not emit_json:
        print()
        print(_header())
        print(_divider())

    for row in rows:
        effective_baseline = baseline_f1 if baseline_f1 is not None else row.avg_f1

        # MetricsSummaryHistoryORM does not store trust_score; recompute
        # it from the stored signals using the current TrustScoreConfig.
        # This is correct for replay: the config at time-of-replay is used,
        # which may differ from the live config at the time of the original
        # aggregation.  That is intentional - replay answers "what would the
        # engine decide *now* given these historical inputs?", not a strict
        # historical reconstruction.
        from model_monitor.monitoring.trust_score import compute_trust_score

        _od_ts = getattr(row, "avg_output_drift_score", None)
        _dq_ts = getattr(row, "avg_data_quality_score", None)
        trust_score, _ = compute_trust_score(
            accuracy=row.avg_accuracy,
            f1=row.avg_f1,
            avg_confidence=row.avg_confidence,
            drift_score=row.avg_drift_score,
            decision_latency_ms=row.avg_latency_ms,
            output_drift_score=_od_ts,
            data_quality_score=_dq_ts,
            config=cfg.trust_score,
        )

        decision = engine.decide(
            batch_index=row.n_batches,
            trust_score=trust_score,
            f1=row.avg_f1,
            f1_baseline=effective_baseline,
            drift_score=row.avg_drift_score,
            recent_actions=recent_actions,
            candidate_exists=candidate_exists,
        )

        if not dry_run:
            try:
                decision_store.record(
                    decision=decision,
                    batch_index=row.n_batches,
                    trust_score=trust_score,
                    f1=row.avg_f1,
                    drift_score=row.avg_drift_score,
                )
            except Exception as exc:
                print(
                    f"warning: failed to persist decision at {_fmt_ts(row.timestamp)} - {exc}",
                    file=sys.stderr,
                )

        # Add decision to recent_actions so subsequent rows see it in cooldown.
        recent_actions = [decision.action, *recent_actions][:10]

        _od = getattr(row, "avg_output_drift_score", None)
        _dq = getattr(row, "avg_data_quality_score", None)
        _cov = getattr(row, "avg_conformal_coverage", None)

        record: dict[str, Any] = {
            "timestamp": row.timestamp,
            "window": window,
            "trust_score": round(trust_score, 6),
            "f1": round(row.avg_f1, 6),
            "input_drift_score": round(row.avg_drift_score, 6),
            "output_drift_score": round(_od, 6) if _od is not None else None,
            "data_quality_score": round(_dq, 6) if _dq is not None else None,
            "conformal_coverage": round(_cov, 6) if _cov is not None else None,
            "action": decision.action,
            "reason": decision.reason,
            "dry_run": dry_run,
        }
        replayed.append(record)

        if emit_json:
            print(json.dumps(record))
        else:
            print(
                _row(
                    ts=row.timestamp,
                    window=window,
                    trust=trust_score,
                    f1=row.avg_f1,
                    drift=row.avg_drift_score,
                    action=decision.action,
                    reason=decision.reason,
                    no_colour=no_colour,
                    output_drift=_od,
                    data_quality=_dq,
                    conformal_cov=_cov,
                )
            )

    if not emit_json:
        print(_divider())
        n_non_none = sum(1 for r in replayed if r["action"] != "none")
        mode = "dry-run" if dry_run else "written to audit log"
        print(
            f"\n  Replayed {len(replayed)} records, "
            f"{n_non_none} non-trivial actions [{mode}].\n"
        )

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """CLI entry point for ``model-monitor replay``."""
    parser = argparse.ArgumentParser(
        prog="model-monitor replay",
        description=(
            "Replay DecisionEngine decisions over stored metric summaries.\n"
            "Useful for debugging past incidents without re-running the full pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  model-monitor replay\n"
            "  model-monitor replay --from 2025-06-01 --window 1h\n"
            "  model-monitor replay --from 2025-06-01T09:00 --to 2025-06-01T18:00 --dry-run\n"
            "  model-monitor replay --from 2025-06-01 --json | jq 'select(.action != \"none\")'\n"
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_ts",
        metavar="ISO",
        default=None,
        help="Start timestamp (inclusive).  ISO date/datetime.  Default: 24 hours ago.",
    )
    parser.add_argument(
        "--to",
        dest="to_ts",
        metavar="ISO",
        default=None,
        help="End timestamp (inclusive).  ISO date/datetime.  Default: now.",
    )
    parser.add_argument(
        "--window",
        choices=["5m", "1h", "24h"],
        default="1h",
        help="Aggregation window to replay.  Default: 1h.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print decisions without writing to the audit log.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Emit newline-delimited JSON instead of the table.",
    )
    parser.add_argument(
        "--no-colour",
        action="store_true",
        help="Disable ANSI colour codes (useful when piping output).",
    )

    args = parser.parse_args()

    now = time.time()
    from_ts = _parse_iso(args.from_ts, field="from") if args.from_ts else now - 86_400
    to_ts = _parse_iso(args.to_ts, field="to") if args.to_ts else now

    if from_ts >= to_ts:
        print("error: --from must be earlier than --to.", file=sys.stderr)
        sys.exit(2)

    sys.exit(
        _run_replay(
            from_ts=from_ts,
            to_ts=to_ts,
            window=args.window,
            dry_run=args.dry_run,
            emit_json=args.emit_json,
            no_colour=args.no_colour or not sys.stdout.isatty(),
        )
    )
