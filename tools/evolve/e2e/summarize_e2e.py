"""Summarize results/evolve/e2e: per run file the raw points, and every system's
throughput / latency at recall 0.90 and 0.95, interpolated log-linearly in miss rate
between the two measured points around the target (no extrapolation).

  python tools/evolve/e2e/summarize_e2e.py [results/evolve/e2e]
"""

import json
import math
import sys
from pathlib import Path


def load(p: Path) -> list[dict]:
    if p.suffix == ".jsonl":
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return json.loads(p.read_text())


def at(rows: list[dict], target: float, key: str) -> float | None:
    rows = sorted(rows, key=lambda r: r["recall"])
    for lo, hi in zip(rows, rows[1:]):
        if lo["recall"] <= target <= hi["recall"] and hi["recall"] > lo["recall"]:
            m = lambda r: math.log(max(1e-4, 1.0 - r))  # noqa: E731
            f = (m(lo["recall"]) - m(target)) / (m(lo["recall"]) - m(hi["recall"]))
            return lo[key] + f * (hi[key] - lo[key])
    return None


d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/evolve/e2e")
fmt = lambda v: "   n/a" if v is None else f"{v:6.0f}" if v > 50 else f"{v:6.3f}"  # noqa: E731
print(f"{'run':24} {'qps@.90':>7} {'lat@.90':>7} {'pg@.90':>6} {'qps@.95':>7} {'lat@.95':>7} {'pg@.95':>6} {'maxR':>6} {'dram':>5}")
for p in sorted(d.glob("*.json*")):
    rows = load(p)
    dram = max(r.get("dram_mb") or r.get("mem_mb", {}).get("RssAnon", 0) for r in rows)
    vals = [at(rows, t, k) for t in (0.90, 0.95) for k in ("qps", "lat_mean_ms", "pages")]
    print(f"{p.stem:24} " + " ".join(fmt(v) for v in vals) + f" {max(r['recall'] for r in rows):6.4f} {dram:5.0f}")
