"""Extend the DiskANN Docker-overhead check to latency percentiles, open-loop
search, and memory-limited search. See run.py for the batch-QPS baseline this
builds on; same index (BIGANN-10M, R=100/L=100), same P-cores (0-7).

Three subcommands, each comparing "suite" (ann-suite container) against
"native" (same runner + diskannpy build, run on the host):

  latency   serial (per-query timed) closed-loop search: p50/p95/p99/max.
  openloop  Poisson arrival-rate search via run_openloop_search (Python-timed).
  memlimit  closed-loop search under a memory cap: Docker mem_limit (cgroup)
            vs systemd-run --scope -p MemoryMax (same cgroup v2 mechanism,
            outside a container), at num_nodes_to_cache=0 so the cap mostly
            constrains OS page cache, not process RSS.

Usage (from the repo root):
    uv run python tools/docker_overhead/run_diskann_extra.py latency --repeats 3
    uv run python tools/docker_overhead/run_diskann_extra.py openloop --repeats 3
    uv run python tools/docker_overhead/run_diskann_extra.py memlimit --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
INDEX_DIR = Path("/home/isfcr/diskann/datasets/sift10m")
INDEX_PREFIX = "disk_index"
QUERIES_NPY = Path("/home/isfcr/data/sift10m/queries.npy")
GT_NPY = Path("/home/isfcr/data/sift10m/ground_truth.npy")
NATIVE_PY = REPO / ".native/diskann/venv/bin/python"
CPUS = "0-7"
IMAGE_ENV = {"OMP_NUM_THREADS": "4", "MKL_THREADING_LAYER": "INTEL", "PYTHONUNBUFFERED": "1"}
NATIVE_ENV = {**os.environ, **IMAGE_ENV, "PYTHONPATH": str(REPO / "library/algorithms")}
RUNNER_DIR = REPO / "library/algorithms/diskann"


def drop_caches() -> None:
    password = os.environ.get("ANN_SUITE_SUDO_PASSWORD")
    cmd = [
        "sudo",
        "-S" if password else "-n",
        "sh",
        "-c",
        "sync; echo 3 > /proc/sys/vm/drop_caches",
    ]
    subprocess.run(
        cmd, input=f"{password}\n" if password else None, text=True, capture_output=True, check=True
    )


def base_config(ls: int, threads: int, batch_mode: bool, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "index_path": None,  # filled per-arm
        "queries_path": str(QUERIES_NPY),
        "ground_truth_path": str(GT_NPY),
        "k": 10,
        "query_rounds": 1,
        "batch_mode": batch_mode,
        "dimension": 128,
        "metric": "L2",
        "cache_warmup_queries": 0,
        "search_args": {
            "Ls": ls,
            "num_threads": threads,
            "index_prefix": INDEX_PREFIX,
            "vector_dtype": "uint8",
            "beam_width": 2,
            "num_nodes_to_cache": 0,
        },
        **extra,
    }


def run_native(config: dict[str, Any], mem_scope: str | None = None) -> dict[str, Any]:
    config = {**config, "index_path": str(INDEX_DIR)}
    # Cgroup cpuset (systemd scope), not taskset: see cpuset.py docstring. When
    # mem_scope is set, fold MemoryMax into the same scope rather than nesting
    # two systemd-run --scope invocations.
    cmd = ["systemd-run", "--scope", "--user", "-p", f"AllowedCPUs={CPUS}"]
    if mem_scope:
        # systemd wants "2G"/"512M" (capital unit suffix), not Docker's "2g".
        systemd_mem = mem_scope[:-1] + mem_scope[-1].upper()
        cmd += ["-p", f"MemoryMax={systemd_mem}", "-p", "MemorySwapMax=0"]
    cmd += [
        "--",
        str(NATIVE_PY),
        "-m",
        "algorithm.runner",
        "--mode",
        "search",
        "--config",
        json.dumps(config),
    ]
    proc = subprocess.run(
        cmd, cwd=RUNNER_DIR, capture_output=True, text=True, check=True, env=NATIVE_ENV
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_suite_container(config: dict[str, Any], mem_limit: str | None) -> dict[str, Any]:
    config = {**config, "index_path": "/data/index"}
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--shm-size",
        "2g",
        "--security-opt",
        "seccomp=unconfined",
        "--cpuset-cpus",
        CPUS,
        "-v",
        f"{INDEX_DIR}:/data/index",
        "-v",
        f"{QUERIES_NPY.parent}:/data/q",
    ]
    if mem_limit:
        cmd += ["--memory", mem_limit, "--memory-swap", mem_limit]
    config["queries_path"] = f"/data/q/{QUERIES_NPY.name}"
    config["ground_truth_path"] = f"/data/q/{GT_NPY.name}"
    cmd += ["ann-suite/diskann:latest", "--mode", "search", "--config", json.dumps(config)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def cmd_latency(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for rep in range(args.repeats):
        for ls in args.ls:
            for arm, fn in (("suite", run_suite_container), ("native", run_native)):
                drop_caches()
                cfg = base_config(ls, args.threads, batch_mode=False, extra={})
                out = fn(cfg, None) if arm == "native" else fn(cfg, None)
                rows.append(
                    {
                        "test": "latency",
                        "arm": arm,
                        "rep": rep,
                        "Ls": ls,
                        "qps": out["qps"],
                        "recall": out["recall"],
                        "mean_ms": out["mean_latency_ms"],
                        "p50_ms": out.get("p50_latency_ms"),
                        "p95_ms": out.get("p95_latency_ms"),
                        "p99_ms": out.get("p99_latency_ms"),
                        "max_ms": out.get("max_latency_ms"),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
    return rows


def cmd_openloop(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for rep in range(args.repeats):
        for rate in args.rates:
            arrival = {
                "mode": "poisson",
                "rate_qps": rate,
                "num_queries": args.num_queries,
                "num_workers": args.threads,
                "warmup_fraction": 0.1,
                "seed": 12345 + rep,
            }
            for arm, fn in (("suite", run_suite_container), ("native", run_native)):
                drop_caches()
                cfg = base_config(
                    args.ls, args.threads, batch_mode=False, extra={"arrival": arrival}
                )
                out = fn(cfg, None)
                ol = out["open_loop"]
                rows.append(
                    {
                        "test": "openloop",
                        "arm": arm,
                        "rep": rep,
                        "rate_qps": rate,
                        "achieved_qps": ol["achieved_qps"],
                        "recall": ol.get("recall"),
                        "mean_ms": ol["latency_ms"]["mean"],
                        "p50_ms": ol["latency_ms"]["p50"],
                        "p99_ms": ol["latency_ms"]["p99"],
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
    return rows


def cmd_memlimit(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for rep in range(args.repeats):
        for cap in args.caps:
            for arm in ("suite", "native"):
                drop_caches()
                cfg = base_config(args.ls, args.threads, batch_mode=True, extra={})
                out = (
                    run_suite_container(cfg, cap)
                    if arm == "suite"
                    else run_native(cfg, mem_scope=cap)
                )
                rows.append(
                    {
                        "test": "memlimit",
                        "arm": arm,
                        "rep": rep,
                        "cap": cap,
                        "qps": out["qps"],
                        "recall": out["recall"],
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
    return rows


def summarize(rows: list[dict[str, Any]], key: str, group_fields: list[str]) -> str:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(tuple(r[f] for f in group_fields), []).append(r)
    lines = [f"median over repeats; {'/'.join(group_fields)}"]
    for g in sorted(groups):
        by_arm: dict[str, list[dict[str, Any]]] = {}
        for r in groups[g]:
            by_arm.setdefault(r["arm"], []).append(r)
        vals = {a: statistics.median(r[key] for r in rs) for a, rs in by_arm.items()}
        ratio = vals.get("suite", 0) / vals["native"] if vals.get("native") else float("nan")
        lines.append(
            f"{g}: suite={vals.get('suite'):.3g} native={vals.get('native'):.3g} ratio={ratio:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("latency")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--ls", type=int, nargs="+", default=[50, 200])

    p = sub.add_parser("openloop")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--ls", type=int, default=100)
    p.add_argument("--num-queries", type=int, default=5000)
    p.add_argument("--rates", type=float, nargs="+", default=[500, 2000])

    p = sub.add_parser("memlimit")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--ls", type=int, default=100)
    p.add_argument("--caps", type=str, nargs="+", default=["1g", "2g", "4g"])

    args = parser.parse_args()
    out = REPO / "results/docker_overhead" / f"{args.cmd}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out.mkdir(parents=True)

    t0 = time.time()
    rows = {"latency": cmd_latency, "openloop": cmd_openloop, "memlimit": cmd_memlimit}[args.cmd](
        args
    )
    (out / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    key = {"latency": "mean_ms", "openloop": "achieved_qps", "memlimit": "qps"}[args.cmd]
    group_fields = {"latency": ["Ls"], "openloop": ["rate_qps"], "memlimit": ["cap"]}[args.cmd]
    summary = summarize(rows, key, group_fields)
    (out / "summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\n{time.time() - t0:.0f}s, results: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
