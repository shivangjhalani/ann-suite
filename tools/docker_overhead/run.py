"""Measure what ann-suite's Docker packaging costs DiskANN search.

Three arms, each run on the same prebuilt index, queries, ground truth, P-cores and
search parameters, with the OS page cache dropped before every point:

  suite   - `ann-suite run` on configs/docker_overhead_diskann_10m.yaml (container)
  native  - the same DiskANN runner and diskannpy build, run directly on the host
            (built by setup_native_diskannpy.sh)
  cpp     - DiskANN's own C++ search_disk_index at the same commit

suite vs native isolates Docker + ann-suite orchestration; native vs cpp shows the
Python binding/runner cost. Arms are interleaved per repeat so drift hits all three.

Usage (from the repo root, ANN_SUITE_SUDO_PASSWORD set if sudo needs a password):
    uv run python tools/docker_overhead/run.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from cpuset import native_cmd_prefix

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/docker_overhead_diskann_10m.yaml"
INDEX_DIR = Path("/home/isfcr/diskann/datasets/sift10m")
INDEX_PREFIX = "disk_index"
QUERIES_NPY = Path("/home/isfcr/data/sift10m/queries.npy")
GT_NPY = Path("/home/isfcr/data/sift10m/ground_truth.npy")
QUERIES_U8BIN = Path("/home/isfcr/data/bigann/query.u8bin")
CPP_SEARCH = Path("/home/isfcr/diskann/DiskANN/build/apps/search_disk_index")
NATIVE_PY = REPO / ".native/diskann/venv/bin/python"
CPUS = "0-7"
K = 10
BEAM_WIDTH = 2
# Mirrors the DiskANN image's ENV so the native runner sees the same OpenMP/MKL setup.
IMAGE_ENV = {"OMP_NUM_THREADS": "4", "MKL_THREADING_LAYER": "INTEL", "PYTHONUNBUFFERED": "1"}


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


def recall_at_k(ids: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean([len(set(a[:K]) & set(b[:K])) / K for a, b in zip(ids, gt, strict=True)]))


def run_suite(out_dir: Path, n_queries: int) -> list[dict[str, Any]]:
    subprocess.run(
        [
            "uv",
            "run",
            "ann-suite",
            "run",
            "--config",
            str(CONFIG),
            "--output",
            str(out_dir),
            "--log-level",
            "WARNING",
        ],
        cwd=REPO,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    rows = []
    for result in json.loads(next(out_dir.rglob("results.json")).read_text()):
        threads = int(result["algorithm"].rsplit("-T", 1)[1])
        stats = result["algorithm_stats"]
        rows.append(
            {
                "threads": threads,
                "Ls": result["hyperparameters"]["search"]["Ls"],
                "qps": result["quality"]["qps"],
                "recall": result["quality"]["recall"],
                "mean_latency_ms": result["latency"]["mean_ms"],
                "ios_per_query": stats["io_reads"] / n_queries,
            }
        )
    return rows


def run_native(threads: int, ls: int) -> dict[str, Any]:
    config = {
        "index_path": str(INDEX_DIR),
        "queries_path": str(QUERIES_NPY),
        "ground_truth_path": str(GT_NPY),
        "k": K,
        "query_rounds": 1,
        "batch_mode": True,
        "dimension": 128,
        "metric": "L2",
        "cache_warmup_queries": 0,
        "search_args": {
            "Ls": ls,
            "num_threads": threads,
            "index_prefix": INDEX_PREFIX,
            "vector_dtype": "uint8",
            "beam_width": BEAM_WIDTH,
            "num_nodes_to_cache": 0,
        },
    }
    proc = subprocess.run(
        [
            *native_cmd_prefix(CPUS),
            str(NATIVE_PY),
            "-m",
            "algorithm.runner",
            "--mode",
            "search",
            "--config",
            json.dumps(config),
        ],
        cwd=REPO / "library/algorithms/diskann",
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **IMAGE_ENV, "PYTHONPATH": str(REPO / "library/algorithms")},
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return {
        "qps": out["qps"],
        "recall": out["recall"],
        "mean_latency_ms": out["mean_latency_ms"],
        "ios_per_query": out["stats"]["io_reads"] / out["total_queries"],
    }


def run_cpp(threads: int, ls: int, scratch: Path, gt: np.ndarray) -> dict[str, Any]:
    prefix = scratch / f"cpp_T{threads}_L{ls}"
    proc = subprocess.run(
        [
            *native_cmd_prefix(CPUS),
            str(CPP_SEARCH),
            "--data_type",
            "uint8",
            "--dist_fn",
            "l2",
            "--index_path_prefix",
            str(INDEX_DIR / INDEX_PREFIX),
            "--query_file",
            str(QUERIES_U8BIN),
            "-K",
            str(K),
            "-L",
            str(ls),
            "-W",
            str(BEAM_WIDTH),
            "-T",
            str(threads),
            "--num_nodes_to_cache",
            "0",
            "--result_path",
            str(prefix),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # Result row (no --gt_file, since gt10m.bin is in PipeANN's layout): L, beamwidth,
    # QPS, mean latency (us), p99.9 latency (us), mean IOs, mean IO time (us), CPU (s).
    row = next(
        [float(x) for x in line.split()]
        for line in proc.stdout.splitlines()
        if re.fullmatch(r"\s*(\d+(\.\d+)?\s+){7}\d+(\.\d+)?\s*", line)
    )
    raw = np.fromfile(f"{prefix}_{ls}_idx_uint32.bin", dtype=np.uint32)
    n, k = raw[:2].view(np.int32)
    ids = raw[2 : 2 + n * k].reshape(n, k)
    return {
        "qps": row[2],
        "recall": recall_at_k(ids, gt),
        "mean_latency_ms": row[3] / 1000.0,
        "ios_per_query": row[5],
    }


def summarize(rows: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["arm"], r["threads"], r["Ls"]), []).append(r)

    def med(arm: str, t: int, ls: int, key: str) -> float:
        return statistics.median(r[key] for r in groups[(arm, t, ls)])

    lines = [
        "median over repeats; suite/native and cpp/native are QPS ratios. In batch mode the",
        "runner's mean_latency_ms is wall/queries, so compare latency only at T=1.",
        f"{'T':>2} {'Ls':>4} | {'QPS suite':>10} {'native':>10} {'cpp':>10} | "
        f"{'suite/nat':>9} {'cpp/nat':>8} | {'recall s/n/c':>20} | {'IOs s/n/c':>17}",
    ]
    for t, ls in sorted({(k[1], k[2]) for k in groups}):
        q = {a: med(a, t, ls, "qps") for a in ("suite", "native", "cpp")}
        rec = "/".join(f"{med(a, t, ls, 'recall'):.4f}" for a in ("suite", "native", "cpp"))
        ios = "/".join(f"{med(a, t, ls, 'ios_per_query'):.1f}" for a in ("suite", "native", "cpp"))
        lines.append(
            f"{t:>2} {ls:>4} | {q['suite']:>10.0f} {q['native']:>10.0f} {q['cpp']:>10.0f} | "
            f"{q['suite'] / q['native']:>9.3f} {q['cpp'] / q['native']:>8.3f} | {rec:>20} | {ios:>17}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--ls", type=int, nargs="+", default=[10, 20, 30, 50, 100, 200])
    args = parser.parse_args()

    out = REPO / "results/docker_overhead" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True)
    gt = np.load(GT_NPY)
    rows: list[dict[str, Any]] = []
    raw_path = out / "raw.jsonl"

    def record(row: dict[str, Any]) -> None:
        rows.append(row)
        with raw_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    for rep in range(args.repeats):
        t0 = time.time()
        for row in run_suite(out / f"suite_rep{rep}", len(gt)):
            if row["threads"] in args.threads and row["Ls"] in args.ls:
                record({"arm": "suite", "rep": rep, **row})
        for threads in args.threads:
            for ls in args.ls:
                drop_caches()
                record(
                    {
                        "arm": "native",
                        "rep": rep,
                        "threads": threads,
                        "Ls": ls,
                        **run_native(threads, ls),
                    }
                )
                drop_caches()
                record(
                    {
                        "arm": "cpp",
                        "rep": rep,
                        "threads": threads,
                        "Ls": ls,
                        **run_cpp(threads, ls, out, gt),
                    }
                )
        print(f"repeat {rep} done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    summary = summarize(rows)
    (out / "summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
