"""Screening rules for device-state runs, applied before any analysis (protocol sections 2-3).

Two rules flag points to exclude and report:
  * stall: a probe before or after the point shows a drive-wide stall (QD1 p50 above
    STALL_QD1_US or QD64 below STALL_QD64_KIOPS);
  * low outlier: QPS below (1 - tol) x the median of its cell (same system, arm, thread
    count and search point). A stall can fall between the two probes: on the SN5100 one
    such point ran at 0.28x its cell median with both probes normal (2026-10-22).
Only the low side is flagged: on this drive a high point is usually a real fresh state.

    uv run python tools/device_state/screen.py results/probe_vs_qps/2026-10-22/probe_vs_qps.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

STALL_QD1_US = 200.0
STALL_QD64_KIOPS = 50.0
CELL_KEYS = ("system", "arm", "threads", "params")


def stalled(row: dict[str, Any]) -> bool:
    for side in ("probe", "post_probe"):
        qd1, qd64 = row.get(f"{side}_qd1_p50_us"), row.get(f"{side}_qd64_kiops")
        if (qd1 is not None and qd1 > STALL_QD1_US) or (
            qd64 is not None and qd64 < STALL_QD64_KIOPS
        ):
            return True
    return False


def cell(row: dict[str, Any], keys: tuple[str, ...] = CELL_KEYS) -> str:
    return json.dumps([row.get(k) for k in keys], sort_keys=True)


def screen(
    rows: list[dict[str, Any]], tol: float = 0.35, keys: tuple[str, ...] = CELL_KEYS
) -> list[str | None]:
    """Per row: None (keep), "stall" or "low_outlier". The cell median uses non-stalled rows."""
    by_cell: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("qps") and not stalled(r):
            by_cell[cell(r, keys)].append(r["qps"])
    medians = {c: statistics.median(v) for c, v in by_cell.items() if len(v) >= 3}
    flags: list[str | None] = []
    for r in rows:
        med = medians.get(cell(r, keys))
        if stalled(r):
            flags.append("stall")
        elif r.get("qps") and med and r["qps"] < (1 - tol) * med:
            flags.append("low_outlier")
        else:
            flags.append(None)
    return flags


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--tol", type=float, default=0.35)
    ap.add_argument("--keys", default=",".join(CELL_KEYS), help="fields that define a cell")
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.jsonl.open()]
    rows = [r for r in rows if r.get("phase") == "point"]
    for r, flag in zip(rows, screen(rows, args.tol, tuple(args.keys.split(","))), strict=True):
        if flag:
            where = {k: r.get(k) for k in ("trial", "round", "system", "arm", "threads")}
            print(json.dumps({"flag": flag, **where, "qps": r.get("qps")}))
    print(f"{len(rows)} points screened")


if __name__ == "__main__":
    main()
