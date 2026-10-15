"""Measure packaged baselines end to end on their prebuilt indexes, at any thread
count, without touching the scoring frontier (evolve_bench.py baselines merges into
it). Writes one row per search point to results/evolve/e2e/<out>.json: recall, QPS,
latency mean/p50/p99, search CPU per query, pages per query and peak anonymous DRAM.

  uv run python tools/evolve/e2e/run_baselines_e2e.py configs/evolve/baselines/starling_b040.yaml \
      --threads 8 --out starling040-t8 [--only Starling-B0.40] [--sweep '[{"Ls": 25}]']
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evolve_bench as eb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline_config", type=Path)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="algorithm names to keep")
    ap.add_argument("--sweep", help="JSON list replacing every algorithm's search sweep")
    ap.add_argument("--config", type=Path, default=eb.REPO / "configs/evolve/bigann10m.yaml")
    a = ap.parse_args()

    eb._ensure_sudo_env()
    cfg = eb._cfg(a.config)
    floors = eb._floors(cfg)
    bcfg = yaml.safe_load(a.baseline_config.read_text())
    if a.only:
        bcfg["algorithms"] = [x for x in bcfg["algorithms"] if x["name"] in a.only]
    for x in bcfg["algorithms"]:
        x["search"] = {**x["search"], "args": {**x["search"].get("args", {})}}
        x["search"]["args"]["num_threads"] = a.threads
        if a.sweep:
            x["search"]["sweep"] = json.loads(a.sweep)
        if not x.get("build", {}).get("prebuilt_path"):
            x["build"] = {**x.get("build", {}), "prebuilt_path": str(eb._built_index(bcfg, x))}
    images = {x["name"]: x["docker_image"] for x in bcfg["algorithms"]}
    for x in bcfg["algorithms"]:
        x.pop("x-system", None)
    for k in [k for k in bcfg if k.startswith("x-")]:
        bcfg.pop(k)
    with eb._evaluation_lock(cfg):
        results = eb._run_suite(bcfg)
    rows = []
    for r in results:
        if not (r.search_result and r.search_result.success) or r.recall is None:
            continue
        base = r.algorithm.split("@")[0]
        out = r.search_result.output or {}
        nq = int(out.get("total_queries") or 0) or 1
        st = r.algorithm_stats
        pages = (
            st.io_reads / nq if st is not None and st.io_reads else r.disk_io.search_pages_per_query
        )
        lat = r.latency
        rows.append(
            {
                "algorithm": base,
                "search": (r.hyperparameters or {}).get("search", {}),
                "threads": a.threads,
                "recall": r.recall,
                "qps": r.qps,
                "lat_mean_ms": getattr(lat, "mean_ms", None),
                "lat_p50_ms": getattr(lat, "p50_ms", None),
                "lat_p99_ms": getattr(lat, "p99_ms", None),
                "cpu_ms": getattr(r.cpu, "search_cpu_time_per_query_ms", None),
                "pages": pages,
                "dram_mb": (r.memory.search_peak_anon_mb or 0.0) - floors.get(images[base], 0.0),
            }
        )
    dst = eb.REPO / cfg["results_dir"] / "e2e" / f"{a.out}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rows, indent=1, default=str))
    print(json.dumps({"rows": len(rows), "out": str(dst)}))


if __name__ == "__main__":
    main()
