"""Rankings of systems at a recall target per (arm, round, threads), from rank_by_state.jsonl.

Each system's two bracketing points are log-miss interpolated to --recall (never extrapolated; a
system whose points do not bracket the target is dropped for that cell). Metric: mean latency at 1
thread (lower is better) and QPS at 8 threads (higher is better). Reports
  * the ranking per arm and round,
  * "fresh vs folded": in each round the Kendall tau between the rankings of the two arms, and
    how many system pairs swap order,
  * the control: the same pair-swap count between two different rounds of the folded arm alone,
  * the share of points whose probes are in the fresh band (QD1 <= 33 us, QD64 >= 500 kIOPS).
  python tools/device_state/analyze_rank_by_state.py results/rank_by_state/DIR [--recall 0.95]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


def interp(pts: list[dict], target: float, key: str) -> float | None:
    pts = sorted(pts, key=lambda r: r["recall"])
    for lo, hi in itertools.pairwise(pts):
        if lo["recall"] <= target <= hi["recall"] and hi["recall"] > lo["recall"]:

            def m(r: float) -> float:
                return math.log(max(1e-4, 1 - r))

            f = (m(lo["recall"]) - m(target)) / (m(lo["recall"]) - m(hi["recall"]))
            return lo[key] + f * (hi[key] - lo[key])
    return None


def swaps(a: dict[str, float], b: dict[str, float]) -> tuple[int, int]:
    common = sorted(set(a) & set(b))
    n = s = 0
    for x, y in itertools.combinations(common, 2):
        n += 1
        s += (a[x] - a[y]) * (b[x] - b[y]) < 0
    return s, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--recall", type=float, default=0.95)
    a = ap.parse_args()
    rows = [json.loads(x) for x in (a.dir / "rank_by_state.jsonl").read_text().splitlines()]
    pts = [r for r in rows if r.get("phase") == "point"]
    fresh_ok = [
        (r["probe_qd1_p50_us"] or 1e9) <= 33 and (r["probe_qd64_kiops"] or 0) >= 500
        for r in pts
        if r["arm"] == "fresh"
    ]
    print(
        f"{len(pts)} points; fresh-arm points whose pre-probe is in the fresh band: {sum(fresh_ok)}/{len(fresh_ok)}"
    )
    cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in pts:
        cell[(r["arm"], r["round"], r["threads"], r["system"])].append(r)
    val: dict[tuple, float] = {}  # (arm, round, threads, system) -> metric
    for (arm, rnd, th, sysn), p in cell.items():
        v = interp(p, a.recall, "mean_latency_ms" if th == 1 else "qps")
        if v is not None:
            val[arm, rnd, th, sysn] = v
    for th, label, sign in (
        (1, "1-thread mean latency ms (lower better)", 1),
        (8, "8-thread QPS (higher better)", -1),
    ):
        print(f"\n== {label} at recall {a.recall}")
        rounds = sorted({k[1] for k in val if k[2] == th})
        by = {(arm, rnd): {k[3]: sign * v for k, v in val.items() if k[0] == arm and k[1] == rnd and k[2] == th}
              for arm in ("fresh", "folded") for rnd in rounds}  # fmt: skip
        for rnd in rounds:
            for arm in ("fresh", "folded"):
                d = by[arm, rnd]
                order = " < ".join(f"{s}({abs(d[s]):.3g})" for s in sorted(d, key=d.get))
                print(f"  r{rnd} {arm:6}: {order}")
        cross = [swaps(by["fresh", r], by["folded", r]) for r in rounds]
        ctrl = [
            swaps(by["folded", r1], by["folded", r2])
            for r1, r2 in itertools.combinations(rounds, 2)
        ]
        print(
            f"  pair swaps fresh-vs-folded within a round: {sum(s for s, _ in cross)}/{sum(n for _, n in cross)}"
        )
        print(
            f"  control, folded round-vs-round:             {sum(s for s, _ in ctrl)}/{sum(n for _, n in ctrl)}"
        )


if __name__ == "__main__":
    main()
