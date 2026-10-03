"""ANN Suite runner for PageANN and LAANN (https://github.com/Dingyi-Kang/PageANN).

Both systems ship as C++ binaries in one DiskANN fork; build_args.variant picks
"pageann" (page graph with inline PQ for uncached neighbors) or "laann" (page
graph + look-ahead search + frequency-sorted pages).

Build pipeline (mirrors upstream sift1M/build_{pageann,laann}_sift1m.sh):
  1. build_vamana_disk_index        (-R/-L/-B/-M; B sets num PQ chunks)
  2. generate_page_graph | generate_laann_graph
  3. build_pageann_nav_graph | build_laann_nav_graph
  4. (laann) reorder_pages_by_frequency  -> *_fsort index

Vector IDs: the page graph renumbers vectors (vectors are regrouped into
pages; a result ID is page * capacity + slot) and LAANN's frequency sort
renumbers pages again (its old_to_new map is already relative to the original
IDs). The binaries return renumbered IDs and do not map them back, so the build
saves the original -> final map (orig_to_final.u32) and search remaps the ground
truth through it before handing it to search_disk_index.

I/O accounting: search_disk_index reads one 4 KB sector per counted IO with
O_DIRECT, so "Mean IOs" is the per-query device page count; it is reported as
stats.io_reads (totals over the run) like the PipeANN runner.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

METRIC_MAP = {"L2": "l2", "euclidean": "l2", "IP": "mips", "inner_product": "mips"}
MANIFEST = "pageann_manifest.json"


def _now() -> str:
    # The image runs Python 3.10 (ubuntu:jammy), which has no datetime.UTC.
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def _tool(name: str) -> str:
    base = Path(os.environ.get("PAGEANN_BIN", "/opt/pageann/build/apps"))
    for cand in (base / name, base / "utils" / name):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"PageANN binary not found: {name}")


def _type_for(dtype: np.dtype) -> str:
    if dtype == np.uint8:
        return "uint8"
    if dtype == np.int8:
        return "int8"
    return "float"


def _write_bin(path: Path, data: np.ndarray) -> str:
    """Write DiskANN .bin (uint32 npts, uint32 dim, row-major data) in chunks."""
    element_type = _type_for(data.dtype)
    out_dtype = {"uint8": np.uint8, "int8": np.int8, "float": np.float32}[element_type]
    with path.open("wb") as f:
        np.asarray(data.shape[:2], dtype=np.uint32).tofile(f)
        step = 1_000_000
        for i in range(0, data.shape[0], step):
            np.ascontiguousarray(data[i : i + step], dtype=out_dtype).tofile(f)
    return element_type


def _write_gt(path: Path, ids: np.ndarray) -> None:
    """DiskANN "ids only" truthset: int32 npts, int32 k, uint32 ids."""
    ids = np.ascontiguousarray(ids, dtype=np.uint32)
    with path.open("wb") as f:
        np.asarray(ids.shape, dtype=np.int32).tofile(f)
        ids.tofile(f)


def _read_map(path: Path) -> np.ndarray:
    """Upstream *_ids_map.bin: int32 npts, int32 dim(=1), uint32 values."""
    return np.fromfile(path, dtype=np.uint32, offset=8)


def _run(cmd: list[str], log: Path) -> str:
    print("$ " + " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    if proc.returncode:
        print(proc.stdout[-4000:], proc.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd[0]}")
    return proc.stdout


def _one(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no file matches {pattern}")
    return Path(hits[0])


def run_build(config: dict[str, Any]) -> dict[str, Any]:
    try:
        start = time.perf_counter()
        index_path = Path(config["index_path"])
        index_path.mkdir(parents=True, exist_ok=True)
        log = index_path / "build.log"
        data = np.load(config["dataset_path"], mmap_mode="r")
        npts, dim = data.shape
        metric = METRIC_MAP[config.get("metric", "L2")]
        args = dict(config.get("build_args", {}))
        variant = str(args.get("variant", "pageann")).lower()
        if variant not in ("pageann", "laann"):
            raise ValueError(f"variant must be pageann|laann, got {variant}")
        R = int(args.get("R", 42))
        L = int(args.get("L", 120))
        B = float(args.get("B", 0.1))  # vamana -B: sets the PQ size (chunks = B GiB / N)
        # PageANN's page graph keeps PQ codes of frequently referenced neighbors in
        # DRAM up to mem_budget_gb and stores the rest inline on disk, so PQ size and
        # DRAM are separate knobs (its low-memory mode); default: tied, as upstream.
        mem_budget = float(args.get("mem_budget_gb", B))
        M = float(args.get("M", 16))
        T = int(args.get("num_threads", os.cpu_count() or 8))
        pq_chunks = min(dim, math.floor(B * 1024**3 / npts))

        data_bin = index_path / "data.bin"
        dtype = _write_bin(data_bin, data)
        vam = index_path / "vamana"
        _run(
            [
                _tool("build_vamana_disk_index"),
                "--data_type",
                dtype,
                "--dist_fn",
                metric,
                "--data_path",
                str(data_bin),
                "--index_path_prefix",
                str(vam),
                "-R",
                str(R),
                "-L",
                str(L),
                "-B",
                str(B),
                "-M",
                str(M),
                "-T",
                str(T),
            ],
            log,
        )

        min_deg = int(args.get("min_degree_per_node", R))
        if variant == "pageann":
            _run(
                [
                    _tool("generate_page_graph"),
                    "--data_type",
                    dtype,
                    "--dist_fn",
                    metric,
                    "--data_path",
                    str(data_bin),
                    "--vamana_index_path_prefix",
                    str(vam),
                    "--R",
                    str(R),
                    "--num_PQ_chunks",
                    str(pq_chunks),
                    "--mem_budget_in_GB",
                    str(mem_budget),
                    "--full_ooc",
                    "false",
                    "--min_degree_per_node",
                    str(min_deg),
                ],
                log,
            )
            page_index = _one(f"{vam}_PGD*_PageANN.index")
            prefix = Path(str(page_index)[: -len(".index")])
            _run(
                [
                    _tool("build_pageann_nav_graph"),
                    "--data_type",
                    dtype,
                    "--dist_fn",
                    metric,
                    "--index_file",
                    str(page_index),
                    "--output_prefix",
                    str(prefix),
                    "--samples_per_page",
                    str(int(args.get("nav_samples_per_page", 1))),
                    "--num_sampled_pages",  # 0 = every page gets a nav node
                    str(int(args.get("nav_sampled_pages", 0))),
                    "-R",
                    str(int(args.get("nav_R", 23))),
                    "-L",
                    str(int(args.get("nav_L", 100))),
                    "--alpha",
                    "1.2",
                    "-T",
                    str(T),
                ],
                log,
            )
            final_prefix = prefix
        else:
            _run(
                [
                    _tool("generate_laann_graph"),
                    "--data_type",
                    dtype,
                    "--dist_fn",
                    metric,
                    "--data_path",
                    str(data_bin),
                    "--vamana_index_path_prefix",
                    str(vam),
                    "--R",
                    str(R),
                    "--min_degree_per_node",
                    str(min_deg),
                    "--num_PQ_chunks",
                    str(pq_chunks),
                    "--L",
                    str(int(args.get("grouping_L", 200))),
                    "--fill_L",
                    str(int(args.get("fill_L", 100))),
                ],
                log,
            )
            page_index = _one(f"{vam}_MGD*_LAANN.index")
            prefix = Path(str(page_index)[: -len(".index")])
            _run(
                [
                    _tool("build_laann_nav_graph"),
                    "--data_type",
                    dtype,
                    "--dist_fn",
                    metric,
                    "--laann_disk_index_file",
                    str(page_index),
                    "--samples_per_page",
                    str(int(args.get("nav_samples_per_page", 1))),
                    "-R",
                    str(int(args.get("nav_R", 24))),
                    "-L",
                    str(int(args.get("nav_L", 100))),
                    "--alpha",
                    "1.2",
                    "-T",
                    str(T),
                ],
                log,
            )
            # reorder_pages_by_frequency insists on a truthset to remap; build
            # has no ground truth, so give it a 1-query dummy and remap the real
            # one at search time through orig_to_final instead. The dummy must
            # carry distances: the tool writes gt_dists unconditionally, and an
            # ids-only truthset leaves that pointer null (segfault in its step 9).
            dummy_gt = index_path / "dummy_gt.bin"
            with dummy_gt.open("wb") as f:
                np.asarray([1, 10], dtype=np.int32).tofile(f)
                np.zeros(10, dtype=np.uint32).tofile(f)
                np.zeros(10, dtype=np.float32).tofile(f)
            _run(
                [
                    _tool("reorder_pages_by_frequency"),
                    "--data_type",
                    dtype,
                    "--dist_fn",
                    metric,
                    "--index_path_prefix",
                    str(prefix),
                    "--data_bin",
                    str(data_bin),
                    "--sample_ratio",
                    str(float(args.get("sample_ratio", 0.01))),
                    "--orig_pq_compressed_file",
                    f"{vam}_pq_compressed.bin",
                    "--orig_gt_file",
                    str(dummy_gt),
                ],
                log,
            )
            final_prefix = Path(f"{prefix}_fsort")

        # Original -> final vector IDs. PageANN: the page graph's map. LAANN: the
        # frequency sort reads the page graph's new_to_old map, so its old_to_new
        # output already maps original dataset IDs straight to final IDs.
        if variant == "laann":
            orig_to_new = _read_map(Path(f"{final_prefix}_old_to_new_ids_map.bin"))
        else:
            orig_to_new = _read_map(_one(f"{index_path}/*_original_to_new_ids_map.bin"))
        if orig_to_new.shape[0] != npts or np.unique(orig_to_new).shape[0] != npts:
            raise RuntimeError("composed ID map is not a permutation of the base set")
        orig_to_new.astype(np.uint32).tofile(index_path / "orig_to_final.u32")

        # Search needs only the page index + its side files; drop build inputs.
        data_bin.unlink(missing_ok=True)
        for f in glob.glob(f"{vam}_disk.index"):
            os.unlink(f)
        manifest = {
            "variant": variant,
            "final_prefix": final_prefix.name,
            "data_type": dtype,
            "metric": metric,
            "R": R,
            "L": L,
            "B": B,
            "pq_chunks": pq_chunks,
        }
        (index_path / MANIFEST).write_text(json.dumps(manifest, indent=2))
        index_size = sum(p.stat().st_size for p in index_path.rglob("*") if p.is_file())
        return {
            "status": "success",
            "build_time_seconds": time.perf_counter() - start,
            "index_size_bytes": index_size,
            **manifest,
        }
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error_message": str(exc),
            "build_time_seconds": 0,
            "index_size_bytes": 0,
        }


def _parse_row(stdout: str, L: int, variant: str) -> dict[str, float]:
    """Parse the results row for search list size L.

    PageANN row: L W QPS Lat IO_us CPU_us Lat-IO MeanIOs Hops Nodes/Hop CacheHit% Recall...
    LAANN row:   L QPS Lat IO_us CPU_us IOs Hops CacheHit% Recall...
    (The PageANN header also names a "Pool Pages" column that the row never prints.)
    """
    nfixed = 11 if variant == "pageann" else 8
    for line in stdout.splitlines():
        toks = line.split()
        if len(toks) <= nfixed or toks[0] != str(L):
            continue
        try:
            v = [float(t) for t in toks]
        except ValueError:
            continue
        if variant == "pageann":
            qps, lat, ios, hops = v[2], v[3], v[7], v[8]
        else:
            qps, lat, ios, hops = v[1], v[2], v[5], v[6]
        recall = v[nfixed]
        return {
            "qps": qps,
            "mean_latency_us": lat,
            "mean_ios": ios,
            "mean_hops": hops,
            "recall": recall / 100.0 if recall > 1.0 else recall,
        }
    raise RuntimeError(f"no result row for L={L} in search_disk_index output")


def run_search(config: dict[str, Any]) -> dict[str, Any]:
    try:
        index_path = Path(config["index_path"])
        manifest = json.loads((index_path / MANIFEST).read_text())
        variant = manifest["variant"]
        k = int(config.get("k", 10))
        args = dict(config.get("search_args", {}))
        L = int(args.get("Ls", 100))
        W = int(args.get("beam_width", 5))
        T = int(args.get("num_threads", 8))

        # Scratch files go to /tmp (container overlay), not the index dir, so
        # concurrent searches on one index cannot clobber each other.
        work = Path("/tmp/pageann_search")
        work.mkdir(parents=True, exist_ok=True)
        queries = np.load(config["queries_path"])
        qbin = work / "queries.bin"
        _write_bin(qbin, queries)
        gt_arg = "null"
        if config.get("ground_truth_path"):
            gt = np.load(config["ground_truth_path"])[:, :k].astype(np.int64)
            remap = np.fromfile(index_path / "orig_to_final.u32", dtype=np.uint32)
            _write_gt(work / "gt.bin", remap[gt])
            gt_arg = str(work / "gt.bin")

        cmd = [
            _tool("search_disk_index"),
            "--data_type",
            manifest["data_type"],
            "--dist_fn",
            manifest["metric"],
            "--index_path_prefix",
            str(index_path / manifest["final_prefix"]),
            "--query_file",
            str(qbin),
            "--gt_file",
            gt_arg,
            "-K",
            str(k),
            "-L",
            str(L),
            "-W",
            str(W),
            "-T",
            str(T),
            "--cache_ratio",
            str(float(args.get("cache_ratio", 0.0))),
            "--nav_L",
            str(int(args.get("nav_L", 10 if variant == "pageann" else 100))),
        ]
        if variant == "laann":
            cmd += [
                "--use_laann",
                "--beamwidth_spike_ratio",
                str(float(args.get("spike_ratio", 0.25))),
                "--beam_decay_ratio",
                str(float(args.get("decay_ratio", 0.95))),
                "--retset_capacity_ratio",
                str(float(args.get("retset_ratio", 2.0))),
            ]

        # Like the PipeANN runner, load and search share one process, so the
        # whole invocation is the search window; per-query IOs come from the
        # binary's own counters, which exclude the load.
        q_start = _now()
        t0 = time.perf_counter()
        out = _run(cmd, work / "search.log")
        wall = time.perf_counter() - t0
        q_end = _now()
        row = _parse_row(out, L, variant)
        nq = len(queries)
        mean_lat_ms = row["mean_latency_us"] / 1000.0
        return {
            "status": "success",
            "total_queries": nq,
            "total_time_seconds": nq / row["qps"] if row["qps"] else wall,
            "qps": row["qps"],
            "recall": row["recall"],
            "mean_latency_ms": mean_lat_ms,
            "p50_latency_ms": mean_lat_ms,
            "p95_latency_ms": mean_lat_ms,
            "p99_latency_ms": mean_lat_ms,
            "warmup_duration_seconds": 0.0,
            "warmup_start_timestamp": q_start,
            "warmup_end_timestamp": q_start,
            "load_duration_seconds": 0.0,
            "query_start_timestamp": q_start,
            "query_end_timestamp": q_end,
            "process_wall_seconds": wall,
            "stats": {
                "io_reads": int(round(row["mean_ios"] * nq)),
                "hops": int(round(row["mean_hops"] * nq)),
            },
        }
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error_message": str(exc),
            "total_queries": 0,
            "total_time_seconds": 0,
            "qps": 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "search"], required=True)
    parser.add_argument("--config", required=True)
    ns = parser.parse_args()
    config = json.loads(ns.config)
    result = run_build(config) if ns.mode == "build" else run_search(config)
    print(json.dumps(result))
    results_dir = Path("/results")
    if results_dir.exists():
        (results_dir / "metrics.json").write_text(json.dumps(result))
    raise SystemExit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
