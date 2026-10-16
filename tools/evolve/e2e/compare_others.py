"""Matched-recall table of all systems from run_others_e2e.sh (BIGANN-10M, ~0.45 GB):
1-thread mean latency and 8-thread QPS (rep1 rep2), pages per query and search DRAM.
Our design v2 at 24 B: Pareto points chosen on rep 3, reported from rep 3 and rep 4.
Tail percentiles are left out: the baseline runners label them inconsistently.
  python tools/evolve/e2e/compare_others.py 0.9 0.95 0.97
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_v2 import at, cell, load, pareto, same_params  # noqa: E402

CONFIGS = {
    "diskann": ["DiskANN-B0.32"],
    "pipeann": ["PipeANN-pq32"],
    "pageann": ["PageANN-B0.32", "LAANN-B0.32"],
    "starling": ["Starling-B0.32"],
}


def main() -> None:
    targets = [float(t) for t in sys.argv[1:]]
    o = {(t, r): load(f"v2-b24-bigann-t{t}-r{r}.jsonl") for t in (1, 8) for r in (3, 4)}
    lat_pts, qps_pts = pareto(o[1, 3], "lat_mean_ms", True), pareto(o[8, 3], "qps", False)
    for t in targets:
        print(
            f"== BIGANN-10M recall {t}: 1-thr mean ms (r1 r2) | 8-thr QPS (r1 r2) | pages | DRAM MB"
        )
        lat = [at(lat_pts, t, "lat_mean_ms"), at(same_params(o[1, 4], lat_pts), t, "lat_mean_ms")]
        qps = [at(qps_pts, t, "qps"), at(same_params(o[8, 4], qps_pts), t, "qps")]
        pg = [at(same_params(o[1, 4], lat_pts), t, "pages")]
        dram = (
            o[1, 3][0]["mem_mb"]["RssAnon"] - o[1, 3][0]["mem_mb_start"]["RssAnon"]
            if o[1, 3]
            else None
        )
        print(
            f"   {'ours v2 24 B':16} {cell(lat, 6, 3)} | {cell(qps, 6, 0)} | {cell(pg, 5, 1)} | {cell([dram], 5, 0)}"
        )
        for cf, names in CONFIGS.items():
            for name in names:
                a = [load(f"others-t1-r{r}-{cf}.json", name) for r in (1, 2)]
                b = [load(f"others-t8-r{r}-{cf}.json", name) for r in (1, 2)]
                dr = [max((x["dram_mb"] for x in a[0]), default=None)]
                print(
                    f"   {name:16} {cell([at(x, t, 'lat_mean_ms') for x in a], 6, 3)} | "
                    f"{cell([at(x, t, 'qps') for x in b], 6, 0)} | {cell([at(a[1], t, 'pages')], 5, 1)} | "
                    f"{cell(dr, 5, 0)}"
                )


if __name__ == "__main__":
    main()
