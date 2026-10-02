"""Suite (real `ann-suite run`, Docker) vs native, interleaved, for PipeANN and SPANN on BIGANN-10M.

Unlike run_pipeann_spann.py (raw `docker run`), the suite arm goes through the whole production
path: config loading, container_runner limits, cgroup monitoring. Both arms search the index
built by run_pipeann_spann.py, pinned to cpuset 0-7, page cache dropped before every point.

  closed  closed-loop QPS sweep
  open    Poisson open-loop, rates around and beyond saturation (fixed num_workers on both arms)
  mem     closed-loop under a memory cap (Docker mem_limit+memswap vs systemd MemoryMax, no swap)

    ANN_SUITE_SUDO_PASSWORD=... uv run python tools/docker_overhead/suite_vs_native.py \\
        pipeann open --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import run_pipeann_spann as h
import yaml

WORKERS = 8
OPEN_QUERIES = 5000
CFG = {
    "pipeann": {
        "closed": [{"Ls": v} for v in (20, 50, 100, 150)],
        "open": {"Ls": 100, "rates": [1500, 3000, 6000]},
        "mem": {"Ls": 100, "limits": ["1500M", "4G"]},
        "args": {"index_prefix": "pipeann", "mode": 2, "beam_width": 32, "nbr_type": "pq"},
        "required": ["pipeann_disk.index", "pipeann_pq_compressed.bin"],
        "image": "ann-suite/pipeann:latest",
    },
    "spann": {
        "closed": [{"internal_result_num": v} for v in (32, 64, 96, 160)],
        "mem": {"internal_result_num": 64, "limits": ["1G", "2G", "8G"]},
        "args": {"posting_page_limit": 3},
        "required": ["SPTAGFullList.bin", "HeadIndex"],
        "image": "ann-suite/spann:latest",
    },
}
DATASET = {
    "name": "sift10m",
    "base_path": "sift10m/base.npy",
    "query_path": "sift10m/queries.npy",
    "ground_truth_path": "sift10m/ground_truth.npy",
    "distance_metric": "L2",
    "dimension": 128,
    "point_type": "float32",
    "base_count": 10_000_000,
    "query_count": 10_000,
}


def suite_run(algo: str, point: dict[str, Any], arrival: dict[str, Any] | None, mem: str | None):
    spec = CFG[algo]
    index_dir = h.REPO / ".native/indices" / algo / "bigann10m"
    out = Path(tempfile.mkdtemp(prefix="suite_vs_native_", dir=h.REPO / "results"))
    search: dict[str, Any] = {
        "timeout_seconds": 1800,
        "k": 10,
        "args": {**spec["args"], **h.ALGOS[algo]["search_args"], **point},
    }
    if arrival:
        search["arrival"] = arrival
    resources: dict[str, Any] = {"cpu_affinity": h.CPUS}
    if mem:
        resources["memory_limit"] = mem.lower()
    cfg = {
        "name": "suite-vs-native",
        "data_dir": str(h.DATA_DIR),
        "results_dir": str(out),
        "index_dir": str(h.REPO / "indices"),
        "resources": resources,
        "algorithms": [
            {
                "name": algo,
                "docker_image": spec["image"],
                "algorithm_type": "disk",
                "build": {"prebuilt_path": str(index_dir), "required_files": spec["required"]},
                "search": search,
            }
        ],
        "datasets": [DATASET],
    }
    cfg_path = out / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    subprocess.run(
        [sys.executable, "-m", "ann_suite.cli", "run", "--config", str(cfg_path)],
        check=True, capture_output=True, text=True, cwd=h.REPO,
    )  # fmt: skip
    res = next(out.rglob("results.json"))
    return json.loads(res.read_text())[0]


def native_run(algo: str, point: dict[str, Any], arrival: dict[str, Any] | None, mem: str | None):
    spec = h.ALGOS[algo]
    cfg: dict[str, Any] = {
        "dimension": 128,
        "metric": "L2",
        "k": 10,
        "queries_path": str(h.DATA_DIR / "sift10m/queries.npy"),
        "ground_truth_path": str(h.DATA_DIR / "sift10m/ground_truth.npy"),
        "search_args": {**spec["search_args"], **point},
    }
    if arrival:
        cfg["arrival"] = {**arrival, "warmup_fraction": arrival.get("warmup_fraction", 0.1)}
    index_dir = h.REPO / ".native/indices" / algo / "bigann10m"
    return h.run_native(algo, index_dir, cfg, memory_max=mem)


def metric(mode: str, arm: str, res: dict[str, Any]) -> dict[str, Any]:
    if mode == "open":
        ol = res["open_loop"]
        return {"achieved_qps": ol.get("achieved_qps"), "p99_ms": ol.get("p99_latency_ms")}
    if arm == "suite":
        return {"qps": res["quality"]["qps"], "recall": res["quality"]["recall"]}
    return {"qps": res.get("qps"), "recall": res.get("recall")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("algo", choices=list(CFG))
    parser.add_argument("mode", choices=["closed", "open", "mem"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rates", type=float, nargs="+", help="override open-loop rates")
    args = parser.parse_args()

    spec = CFG[args.algo]
    if args.mode == "open" and "open" not in spec:
        parser.error(f"{args.algo} has no open-loop mode in ann-suite (docs/OPEN_LOOP.md)")
    plan: list[tuple[dict[str, Any], dict[str, Any] | None, str | None]] = []
    if args.mode == "closed":
        plan = [(p, None, None) for p in spec["closed"]]
    elif args.mode == "open":
        base = {k: v for k, v in spec["open"].items() if k != "rates"}
        rates = args.rates or spec["open"]["rates"]
        for r in rates:
            arrival = {"mode": "poisson", "rate_qps": r, "num_queries": OPEN_QUERIES,
                       "num_workers": WORKERS, "seed": 12345}  # fmt: skip
            plan.append((base, arrival, None))
    else:
        base = {k: v for k, v in spec["mem"].items() if k != "limits"}
        plan = [(base, None, m) for m in spec["mem"]["limits"]]

    out = (
        h.REPO
        / "results/docker_overhead"
        / f"suite_{args.algo}_{args.mode}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    out.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for rep in range(args.repeats):
        for point, arrival, mem in plan:
            for arm, fn in (("suite", suite_run), ("native", native_run)):
                h.drop_caches()
                res = fn(args.algo, point, arrival, mem)
                key = json.dumps({**point, **({"rate": arrival["rate_qps"]} if arrival else {}),
                                  **({"mem": mem} if mem else {})})  # fmt: skip
                row = {"rep": rep, "arm": arm, "point": key, **metric(args.mode, arm, res)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    (out / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    invalid = [
        r for r in rows if r.get("recall") == 0 or r.get("recall") is None and args.mode != "open"
    ]
    rows = [r for r in rows if r not in invalid]
    field = "achieved_qps" if args.mode == "open" else "qps"
    lines = [f"{args.algo} {args.mode}: median {field}, suite/native"]
    if invalid:
        lines.append(
            f"DISCARDED {len(invalid)} rows with zero/missing recall (search failed, e.g. OOM)"
        )
    for key in dict.fromkeys(r["point"] for r in rows):
        s = [r[field] for r in rows if r["point"] == key and r["arm"] == "suite" and r[field]]
        n = [r[field] for r in rows if r["point"] == key and r["arm"] == "native" and r[field]]
        if s and n:
            ms, mn = statistics.median(s), statistics.median(n)
            lines.append(f"{key}: suite={ms:.4g} native={mn:.4g} ratio={ms / mn:.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
