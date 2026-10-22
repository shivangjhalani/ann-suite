"""Does tuning MaxDistRatio / head MaxCheck beat an IRN-only sweep? Compare interpolated Pareto frontiers.

For each index, two frontiers over (recall, cost) measured at 1 thread: the default-knob IRN sweep, and the
union of every measured knob combination. Both are piecewise-linear over measured points, so the comparison
is not distorted by the coarse IRN grid. Cost = mean latency, or posting reads per query.
"""

import csv
import itertools
import sys
from pathlib import Path

run = Path(sys.argv[1])
rows = [r for r in csv.DictReader((run / "points.csv").open()) if r.get("recall")]
STAGES = ("A_irn_all_indexes", "B1_max_dist_ratio", "B2_head_maxcheck", "B3_joint_mdr_maxcheck")
INDEXES = ["10m/ratio0.1_rep8_ppl4", "100m/lean_c_ratio0.05_rep8_ppl8"]
TARGETS = (0.90, 0.95, 0.97, 0.98, 0.99)


def frontier(points: list[tuple[float, float, dict]]) -> list[tuple[float, float, dict]]:
    """Non-dominated (recall up, cost down) points, ordered by recall."""
    best, keep = float("inf"), []
    for rec, cost, r in sorted(points, key=lambda p: (-p[0], p[1])):
        if cost < best:
            keep.append((rec, cost, r))
            best = cost
    return sorted(keep, key=lambda p: p[0])


def at(front: list[tuple[float, float, dict]], target: float) -> tuple[float, str] | None:
    for (r0, c0, a), (r1, c1, b) in itertools.pairwise(front):
        if r0 <= target <= r1:
            cost = c0 if r1 == r0 else c0 + (c1 - c0) * (target - r0) / (r1 - r0)
            return cost, f"{label(a)} .. {label(b)}"
    return None


def label(r: dict) -> str:
    parts = [f"{r['irn']}"]
    if r["mdr"] not in ("", "None"):
        parts.append(f"mdr{r['mdr']}")
    if r["maxcheck"] not in ("", "None"):
        parts.append(f"mc{r['maxcheck']}")
    return "/".join(parts)


for index in INDEXES:
    pts = [
        r for r in rows if r["index"] == index and r["stage"] in STAGES and int(r["threads"]) == 1
    ]
    default = [r for r in pts if r["mdr"] in ("", "None") and r["maxcheck"] in ("", "None")]
    print(f"\n### {index}: {len(pts)} measured combinations, {len(default)} with default knobs\n")
    for metric, unit, col in (
        ("mean latency", "ms", "lat_mean_ms"),
        ("posting reads", "reads/query", "ops_q"),
    ):
        fd = frontier([(float(r["recall"]), float(r[col]), r) for r in default])
        fa = frontier([(float(r["recall"]), float(r[col]), r) for r in pts])
        print(f"{metric} ({unit}) to reach a recall target, interpolated\n")
        print(
            "| recall | IRN sweep only | best over all knobs | saved | frontier segment (IRN/mdr/maxcheck) |"
        )
        print("|---|---|---|---|---|")
        for t in TARGETS:
            d, a = at(fd, t), at(fa, t)
            if d is None or a is None:
                print(f"| {t} | n/a | n/a | | |")
                continue
            print(f"| {t} | {d[0]:.3f} | {a[0]:.3f} | {1 - a[0] / d[0]:.0%} | {a[1]} |")
        print()
