"""model-monitor export - write the decision audit log to CSV or JSON.

Reads ``DecisionStore`` (the full per-decision audit log) and writes every
record within an optional time range to stdout or a file.

Usage
-----
::

    model-monitor export                           # CSV to stdout
    model-monitor export --format json             # NDJSON to stdout
    model-monitor export --from 2025-01-01         # filter by date
    model-monitor export --from 2025-01-01T09:00 --to 2025-01-01T18:00
    model-monitor export --output decisions.csv    # write to file
    model-monitor export --format json | jq 'select(.action != "none")'

Exit codes
----------
0  Export completed (even if no records matched).
1  Storage or I/O error.
2  Invalid arguments.

Design notes
------------
The output schema is stable: column order in CSV is fixed and NDJSON field
names match the DecisionRecord ORM columns.  Downstream scripts can rely on
the schema without version-gating.

Both CSV and NDJSON write a header/schema comment so the output is
self-documenting when inspected with ``head`` or ``less``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_monitor.storage.decision_store import DecisionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso(value: str, *, field: str) -> float:
    """Parse an ISO 8601 date or datetime string to a UTC Unix timestamp."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
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


_CSV_FIELDS = [
    "id",
    "timestamp",
    "timestamp_iso",
    "action",
    "reason",
    "batch_index",
    "trust_score",
    "f1",
    "drift_score",
    "model_version",
]


def _row_to_dict(row: Any) -> dict[str, Any]:
    ts = float(row.timestamp)
    return {
        "id": row.id,
        "timestamp": ts,
        "timestamp_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        "action": row.action,
        "reason": row.reason,
        "batch_index": row.batch_index,
        "trust_score": round(float(row.trust_score or 0), 6),
        "f1": round(float(row.f1 or 0), 6),
        "drift_score": round(float(row.drift_score or 0), 6),
        "model_version": row.model_version or "",
    }


# ---------------------------------------------------------------------------
# Core export
# ---------------------------------------------------------------------------


def _run_export(
    *,
    from_ts: float | None,
    to_ts: float | None,
    fmt: str,
    output: Path | None,
) -> int:
    """Fetch records and write them to stdout or a file.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    try:
        store = DecisionStore()
        rows = store.query_range(from_ts=from_ts, to_ts=to_ts)
    except Exception as exc:
        print(f"error: could not read DecisionStore - {exc}", file=sys.stderr)
        return 1

    dest_file: Any = (
        open(output, "w", encoding="utf-8", newline="") if output else sys.stdout
    )

    try:
        if fmt == "csv":
            _write_csv(rows, dest_file)
        else:
            _write_ndjson(rows, dest_file)
    except BrokenPipeError:
        # Piping to head/less - not an error.
        pass
    except Exception as exc:
        print(f"error: write failed - {exc}", file=sys.stderr)
        return 1
    finally:
        if output and dest_file is not sys.stdout:
            dest_file.close()

    if output:
        n = len(rows)
        print(f"Exported {n} record{'s' if n != 1 else ''} → {output}", file=sys.stderr)

    return 0


def _write_csv(rows: list[Any], dest: Any) -> None:
    writer = csv.DictWriter(dest, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(_row_to_dict(row))


def _write_ndjson(rows: list[Any], dest: Any) -> None:
    for row in rows:
        dest.write(json.dumps(_row_to_dict(row)) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """CLI entry point for ``model-monitor export``."""
    parser = argparse.ArgumentParser(
        prog="model-monitor export",
        description="Export the decision audit log to CSV or NDJSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  model-monitor export\n"
            "  model-monitor export --from 2025-06-01 --output decisions.csv\n"
            "  model-monitor export --format json | jq 'select(.action != \"none\")'\n"
            "  model-monitor export --from 2025-06-01T09:00 --to 2025-06-01T18:00\n"
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_ts",
        metavar="ISO",
        default=None,
        help="Start timestamp (inclusive). ISO date/datetime. Default: all records.",
    )
    parser.add_argument(
        "--to",
        dest="to_ts",
        metavar="ISO",
        default=None,
        help="End timestamp (inclusive). ISO date/datetime. Default: now.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format: 'csv' (default) or 'json' (newline-delimited).",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        default=None,
        help="Write to FILE instead of stdout. Format inferred from extension if --format not given.",
    )

    args = parser.parse_args()

    from_ts = _parse_iso(args.from_ts, field="from") if args.from_ts else None
    to_ts = _parse_iso(args.to_ts, field="to") if args.to_ts else None

    if from_ts is not None and to_ts is not None and from_ts >= to_ts:
        print("error: --from must be earlier than --to.", file=sys.stderr)
        sys.exit(2)

    fmt = args.format
    # Infer format from output file extension when --format not explicitly given
    if args.output and args.format == "csv":
        ext = Path(args.output).suffix.lower()
        if ext in (".json", ".ndjson", ".jsonl"):
            fmt = "json"

    sys.exit(
        _run_export(
            from_ts=from_ts,
            to_ts=to_ts,
            fmt=fmt,
            output=Path(args.output) if args.output else None,
        )
    )
