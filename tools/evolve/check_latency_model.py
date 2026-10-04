"""Check score v2's latency model against measured latency (docs/EVOLVE_SCORE.md).

For every baseline point measured on one search thread whose I/O rounds are known
(hops per query: DiskANN, PageANN, LAANN), the model predicts
    hops * d(pages / hops) + cpu
and this compares it with the measured mean latency. Prints one line per point and
the median and worst ratio per system; the model is used only if every system's
median is within the tolerance (default 25%).

  .venv/bin/python tools/evolve/check_latency_model.py [--tolerance 0.25]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evolve_bench import REPO, _cfg, _host  # noqa: E402
from frontier import load_points  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs/evolve/bigann10m.yaml")
    ap.add_argument("--tolerance", type=float, default=0.25)
    ns = ap.parse_args()
    cfg = _cfg(ns.config)
    host = _host(cfg)
    if host is None:
        raise SystemExit("no host model: run calibrate_ssd.py first")
    ratios: dict[str, list[float]] = defaultdict(list)
    for p in load_points(REPO / cfg["frontier"]):
        hops, lat, cpu = (p.extra.get(k) for k in ("hops_per_query", "latency_ms", "cpu_ms"))
        if not hops or lat is None or cpu is None or p.extra.get("search_threads") != 1:
            continue
        model = hops * host.round_ms(p.pages / hops) + cpu
        ratios[p.system].append(model / lat)
        print(
            f"{p.label[:60]:60} pages {p.pages:6.1f} hops {hops:5.1f} cpu {cpu:.3f} | "
            f"measured {lat:.3f} ms, model {model:.3f} ms, ratio {model / lat:.2f}"
        )
    ok = bool(ratios)
    summary = {}
    for system, r in sorted(ratios.items()):
        med = statistics.median(r)
        worst = max(r, key=lambda x: abs(x - 1))
        summary[system] = {"points": len(r), "median_ratio": med, "worst_ratio": worst}
        ok &= abs(med - 1) <= ns.tolerance
    print(json.dumps({"ok": ok, "tolerance": ns.tolerance, "systems": summary}))


if __name__ == "__main__":
    main()
