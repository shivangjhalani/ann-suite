"""Matched-recall comparison of our design with Starling from results/evolve/e2e:
1-thread mean latency, 8-thread QPS and pages per query at each recall target,
both repetitions side by side (log-miss interpolation between the two measured
points around the target; nothing is extrapolated).

  python tools/evolve/e2e/compare_e2e.py DATASET TARGET [TARGET ...]   # DATASET: deep | t2i
"""

import json
import math
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[3] / "results/evolve/e2e"


def load(name: str, algo: str | None) -> list[dict]:
    p = D / name
    rows = (
        [json.loads(x) for x in p.read_text().splitlines()]
        if name.endswith("l")
        else json.loads(p.read_text())
    )
    return [r for r in rows if algo is None or r["algorithm"] == algo]


def at(rows: list[dict], target: float, key: str) -> float | None:
    rows = sorted(rows, key=lambda r: r["recall"])
    for lo, hi in zip(rows, rows[1:], strict=False):
        if lo["recall"] <= target <= hi["recall"] and hi["recall"] > lo["recall"]:

            def m(r: float) -> float:
                return math.log(max(1e-4, 1 - r))

            f = (m(lo["recall"]) - m(target)) / (m(lo["recall"]) - m(hi["recall"]))
            return lo[key] + f * (hi[key] - lo[key])
    return None


def cell(vals: list[float | None], width: int, prec: int) -> str:
    return " ".join(f"{'n/a':>{width}}" if v is None else f"{v:{width}.{prec}f}" for v in vals)


ds, targets = sys.argv[1], [float(t) for t in sys.argv[2:]]
for t in targets:
    print(
        f"== {ds} recall {t}:  1-thread latency ms (rep1 rep2) | 8-thread QPS (rep1 rep2) | pages"
    )
    for name in ["ours", f"Starling-B0.32-{ds}", f"Starling-B0.40-{ds}"]:
        lat, qps, pg = [], [], []
        for rep in (1, 2):
            if name == "ours":
                a = load(f"ours-g40-{ds}-t1-r{rep}.jsonl", None)
                b = load(f"ours-g40-{ds}-t8-r{rep}.jsonl", None)
            else:
                a = load(f"starling-{ds}-t1-r{rep}.json", name)
                b = load(f"starling-{ds}-t8-r{rep}.json", name)
            lat.append(at(a, t, "lat_mean_ms"))
            qps.append(at(b, t, "qps"))
            pg.append(at(a, t, "pages"))
        print(f"   {name:20} {cell(lat, 6, 3)} | {cell(qps, 6, 0)} | {cell(pg, 5, 1)}")
