"""ANN Suite runner for PipeANN (https://github.com/thustorage/PipeANN).

PipeANN ships as C++ binaries (no Python bindings), so this runner shells out
to the tools built into the image (see the Dockerfile): `build_disk_index`,
`build_memory_index`, `gen_random_slice`, `search_disk_index` for closed-loop
search, and `search_openloop` for the open-loop (arrival-rate) search mode.

Vector dtype: unlike the DiskANN/SPANN runners, this runner does NOT take a
separate `vector_dtype`/`point_type` config knob that has to be kept in sync
with the dataset by hand. It infers the PipeANN element type directly from
the loaded .npy array's dtype (uint8 -> "uint8", int8 -> "int8", anything
else -> "float", after casting to float32). That sidesteps a class of
mismatch bugs the other runners are exposed to when a config's declared
`point_type`/`vector_dtype` drifts from the dataset's actual on-disk dtype.

Binary formats: PipeANN's own .bin format (uint32 num_points, uint32 dim,
then flattened row-major vectors) for base/query data, and DiskANN-compatible
ground truth (int32 npts, int32 k, then npts*k uint32 ids - the "ids only",
no-distances variant `load_truthset` also accepts).

Index files are written to /data/index/ (the suite's container_index_path)
under a configurable prefix (`index_prefix`, default "pipeann"), so the host
monitor's I/O metrics for the search phase capture real device reads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

METRIC_MAP = {
    "L2": "l2",
    "euclidean": "l2",
    "IP": "mips",
    "inner_product": "mips",
    "cosine": "cosine",
    "angular": "cosine",
}

# search_disk_index's results table header, e.g.:
#      L   I/O Width         QPS  AvgLat(us)     P99 Lat    Mean IOs   Recall@10
#     10          32     1871.92      512.01      939.00       23.24       67.40
_RESULT_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)

# search_openloop's single summary line, e.g.:
# RESULT mode=2 T=8 W=32 L=50 lambda=1000.0 achieved_qps=943.2 recall=0.9421
# lat_mean=812.3 p50=701.0 p90=1502.0 p99=3011.0 p999=8820.5 svc_mean=650.1
# svc_p99=2200.0 ios=41.2 dev_iops=15342.0 dev_util=0.812 cpu=0.734 dev_lat=52.1
_OPENLOOP_KV_RE = re.compile(r"(\w+)=([-\d.]+)")


def _pipeann_type_for(dtype: np.dtype) -> str:
    """Map a numpy dtype to PipeANN's <type> CLI argument."""
    if dtype == np.uint8:
        return "uint8"
    if dtype == np.int8:
        return "int8"
    return "float"


def _numpy_dtype_for(pipeann_type: str) -> np.dtype:
    return {
        "uint8": np.dtype(np.uint8),
        "int8": np.dtype(np.int8),
        "float": np.dtype(np.float32),
    }[pipeann_type]


def _write_bin(path: Path, data: np.ndarray) -> str:
    """Write `data` in PipeANN's .bin format; return the resolved element type."""
    element_type = _pipeann_type_for(data.dtype)
    array = data if data.dtype == _numpy_dtype_for(element_type) else data.astype(np.float32)
    with path.open("wb") as f:
        np.asarray([array.shape[0], array.shape[1]], dtype=np.uint32).tofile(f)
        array.tofile(f)
    return element_type


def _write_gt_bin(path: Path, ground_truth: np.ndarray, k: int) -> None:
    """Write ground truth in PipeANN/DiskANN's "ids only" truthset format."""
    ids = ground_truth[:, :k].astype(np.uint32)
    with path.open("wb") as f:
        np.asarray([ids.shape[0], ids.shape[1]], dtype=np.int32).tofile(f)
        ids.tofile(f)


def _metric_arg(metric: str) -> str:
    if metric not in METRIC_MAP:
        raise ValueError(f"Unsupported PipeANN metric: {metric}")
    return METRIC_MAP[metric]


def _bin_dir() -> Path:
    return Path(os.environ.get("PIPEANN_BIN", "/opt/pipeann/build/tests"))


def _util_bin_dir() -> Path:
    return Path(os.environ.get("PIPEANN_UTIL_BIN", "/opt/pipeann/build/tests/utils"))


def _tool(name: str, *, util: bool = False) -> str:
    return str((_util_bin_dir() if util else _bin_dir()) / name)


def _run(
    command: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a PipeANN binary and preserve its stdout/stderr for diagnostics."""
    full_env = {**os.environ, **(env or {})}
    result = subprocess.run(command, text=True, capture_output=True, check=False, env=full_env)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode:
        print(result.stdout, file=sys.stderr, end="")
        raise RuntimeError(f"PipeANN command failed ({result.returncode}): {' '.join(command)}")
    return result


def _arrival_env(search_args: dict[str, Any]) -> dict[str, str]:
    """Env vars for the io_governor / SQPOLL knobs (unset = stock PipeANN)."""
    env: dict[str, str] = {}
    if search_args.get("sqpoll", True) is False:
        env["PIPEANN_SQPOLL"] = "0"
    if "qstar" in search_args and search_args["qstar"] is not None:
        env["PIPEANN_QSTAR"] = str(int(search_args["qstar"]))
    if "wmin" in search_args and search_args["wmin"] is not None:
        env["PIPEANN_WMIN"] = str(int(search_args["wmin"]))
    return env


def run_build(config: dict[str, Any]) -> dict[str, Any]:
    try:
        dataset_path = Path(config["dataset_path"])
        data = np.load(dataset_path)
        print(f"Loaded {len(data)} vectors ({data.dtype}) from {dataset_path}", file=sys.stderr)

        index_path = Path(config["index_path"])
        index_path.mkdir(parents=True, exist_ok=True)
        metric = _metric_arg(config.get("metric", "L2"))
        build_args = dict(config.get("build_args", {}))

        index_prefix = str(build_args.get("index_prefix", "pipeann"))
        prefix_path = index_path / index_prefix

        data_bin = index_path / "data.bin"
        element_type = _write_bin(data_bin, data)

        R = int(build_args.get("R", 96))
        L = int(build_args.get("L", 128))
        pq_bytes = int(build_args.get("pq_bytes", 32))
        build_memory_gb = int(build_args.get("build_memory_gb", 64))
        threads = int(build_args.get("num_threads", os.cpu_count() or 8))
        nbr_type = str(build_args.get("nbr_type", "pq"))  # pq | rabitq | rabitq{3-5}
        builder = str(build_args.get("builder", "vamana"))  # vamana | pipnn
        L2 = int(build_args.get("L2", 0))

        cmd = [
            _tool("build_disk_index"),
            element_type,
            str(data_bin),
            str(prefix_path),
            str(R),
            str(L),
            str(pq_bytes),
            str(build_memory_gb),
            str(threads),
            metric,
            nbr_type,
        ]
        if builder == "pipnn":
            if L2 <= 0:
                raise ValueError("build_args.L2 must be > 0 when builder='pipnn'")
            cmd.append(str(L2))

        start = time.perf_counter()
        _run(cmd)

        # Optional in-memory entry-point index, built from a random sample.
        # Boosts search for lower-dimensional datasets (SIFT/DEEP/SPACEV per
        # upstream docs); skip with build_args.build_mem_index: false or by
        # setting search_args.mem_L: 0 at search time regardless.
        built_mem_index = False
        if build_args.get("build_mem_index", True):
            sample_rate = float(build_args.get("mem_sample_rate", 0.01))
            sample_prefix = index_path / f"{index_prefix}_SAMPLE_RATE_{sample_rate}"
            _run(
                [
                    _tool("gen_random_slice", util=True),
                    element_type,
                    str(data_bin),
                    str(sample_prefix),
                    str(sample_rate),
                ]
            )
            mem_R = int(build_args.get("mem_R", 32))
            mem_L = int(build_args.get("mem_L", 64))
            mem_alpha = float(build_args.get("mem_alpha", 1.2))
            mem_threads = int(build_args.get("mem_threads", threads))
            _run(
                [
                    _tool("build_memory_index"),
                    element_type,
                    f"{sample_prefix}_data.bin",
                    f"{sample_prefix}_ids.bin",
                    f"{prefix_path}_mem.index",
                    str(mem_R),
                    str(mem_L),
                    str(mem_alpha),
                    str(mem_threads),
                    metric,
                ]
            )
            built_mem_index = True

        build_time = time.perf_counter() - start
        index_size = sum(p.stat().st_size for p in index_path.rglob("*") if p.is_file())

        return {
            "status": "success",
            "build_time_seconds": build_time,
            "index_size_bytes": index_size,
            "index_prefix": index_prefix,
            "element_type": element_type,
            "mem_index_built": built_mem_index,
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


def _prepare_queries_and_gt(
    config: dict[str, Any], index_path: Path, k: int
) -> tuple[Path, Path | None, str, int]:
    queries = np.load(Path(config["queries_path"]))
    element_type = _pipeann_type_for(queries.dtype)
    queries_bin = index_path / "queries.bin"
    _write_bin(queries_bin, queries)

    gt_bin_path: Path | None = None
    gt_path = config.get("ground_truth_path")
    if gt_path and Path(gt_path).exists():
        ground_truth = np.load(Path(gt_path))
        gt_bin_path = index_path / "gt.bin"
        _write_gt_bin(gt_bin_path, ground_truth, k)

    return queries_bin, gt_bin_path, element_type, len(queries)


def run_search(config: dict[str, Any]) -> dict[str, Any]:
    try:
        index_path = Path(config["index_path"])
        k = int(config.get("k", 10))
        metric = _metric_arg(config.get("metric", "L2"))
        search_args = dict(config.get("search_args", {}))
        index_prefix = str(search_args.get("index_prefix", "pipeann"))
        prefix_path = index_path / index_prefix

        queries_bin, gt_bin, element_type, num_queries = _prepare_queries_and_gt(
            config, index_path, k
        )
        gt_arg = str(gt_bin) if gt_bin is not None else "null"

        mode = int(search_args.get("mode", 2))  # 0 beam-search, 2 PipeANN (recommended), 3 coro
        nbr_type = str(search_args.get("nbr_type", "pq"))
        beam_width = int(search_args.get("beam_width", 32))
        mem_L = int(search_args.get("mem_L", 0))
        Ls = int(search_args.get("Ls", 100))
        num_threads = int(search_args.get("num_threads", 1))
        env = _arrival_env(search_args)

        arrival = config.get("arrival")

        # PipeANN's search binaries load the index and run the timed search in
        # a single process invocation with no separate load/search signal we
        # can observe from the outside. Unlike DiskANN (diskannpy separates
        # load() and search() as distinct Python calls), there is no way to
        # split warmup (index load) from search here without patching the
        # binaries to emit a timestamp between the two - so, like the SPANN
        # runner, treat the whole invocation as the SEARCH window (the
        # dominant cost for a disk index) and report zero warmup, rather than
        # mislabeling real search I/O as "warmup" (or worse, collapsing the
        # search window to ~0s).
        query_start_timestamp = datetime.now(UTC).isoformat()
        search_start = time.perf_counter()

        if arrival:
            result = _run_openloop(
                prefix_path,
                element_type,
                num_threads,
                beam_width,
                queries_bin,
                gt_arg,
                k,
                metric,
                nbr_type,
                mode,
                mem_L,
                Ls,
                arrival,
                env,
            )
        else:
            result = _run_closedloop(
                prefix_path,
                element_type,
                num_threads,
                beam_width,
                queries_bin,
                gt_arg,
                k,
                metric,
                nbr_type,
                mode,
                mem_L,
                Ls,
                num_queries,
                int(config.get("query_rounds", 1)),
                env,
            )

        search_wall_seconds = time.perf_counter() - search_start
        query_end_timestamp = datetime.now(UTC).isoformat()

        result.update(
            {
                "status": "success",
                "warmup_duration_seconds": 0.0,
                "warmup_start_timestamp": query_start_timestamp,
                "warmup_end_timestamp": query_start_timestamp,
                "load_duration_seconds": 0.0,
                "query_start_timestamp": query_start_timestamp,
                "query_end_timestamp": query_end_timestamp,
                "cache_warmup_queries_requested": 0,
                "cache_warmup_queries_executed": 0,
                "cache_warmup_duration_seconds": 0.0,
            }
        )
        # search_wall_seconds includes index load; total_time_seconds from the
        # binary's own self-timed table/RESULT line is the pure query time and
        # is what QPS/latency are computed from. Keep both for diagnostics.
        result.setdefault("process_wall_seconds", search_wall_seconds)
        return result
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


def _run_closedloop(
    prefix_path: Path,
    element_type: str,
    num_threads: int,
    beam_width: int,
    queries_bin: Path,
    gt_arg: str,
    k: int,
    metric: str,
    nbr_type: str,
    mode: int,
    mem_L: int,
    Ls: int,
    num_queries: int,
    query_rounds: int,
    env: dict[str, str],
) -> dict[str, Any]:
    cmd = [
        _tool("search_disk_index"),
        element_type,
        str(prefix_path),
        str(num_threads),
        str(beam_width),
        str(queries_bin),
        gt_arg,
        str(k),
        metric,
        nbr_type,
        str(mode),
        str(mem_L),
        str(Ls),
    ]

    rows: list[tuple[float, ...]] = []
    total_time = 0.0
    for _ in range(max(1, query_rounds)):
        start = time.perf_counter()
        proc = _run(cmd, env=env)
        total_time += time.perf_counter() - start
        for line in proc.stdout.splitlines():
            m = _RESULT_ROW_RE.match(line)
            if m:
                rows.append(tuple(float(g) for g in m.groups()))

    if not rows:
        raise RuntimeError(
            "search_disk_index produced no parseable result row; stdout did not "
            "match the expected 'L I/O-Width QPS AvgLat P99Lat MeanIOs Recall' table"
        )

    _l, _io_width, qps, avg_lat_us, p99_lat_us, mean_ios, recall_pct = rows[-1]
    total_queries = num_queries * max(1, query_rounds)

    return {
        "total_queries": total_queries,
        "total_time_seconds": total_time,
        "qps": qps,
        "recall": recall_pct / 100.0,
        "mean_latency_ms": avg_lat_us / 1000.0,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "p99_latency_ms": p99_lat_us / 1000.0,
        "max_latency_ms": None,
        "stats": {"io_reads": int(round(mean_ios * max(1, total_queries)))},
    }


def _run_openloop(
    prefix_path: Path,
    element_type: str,
    default_num_threads: int,
    beam_width: int,
    queries_bin: Path,
    gt_arg: str,
    k: int,
    metric: str,
    nbr_type: str,
    mode: int,
    mem_L: int,
    Ls: int,
    arrival: dict[str, Any],
    env: dict[str, str],
) -> dict[str, Any]:
    """Run the open-loop (arrival-rate) search via the search_openloop binary.

    See vendor/search_openloop.cpp: a fixed worker-thread pool serves a FIFO
    queue of queries scheduled on a Poisson (or, for mode="closed", all-at-t=0)
    arrival process. Reported latency is completion - scheduled arrival time
    (queueing included); the first 10% of issued queries are dropped from the
    reported percentiles by the driver itself as steady-state warm-up.
    """
    num_workers = int(arrival.get("num_workers") or default_num_threads)
    num_queries = int(arrival.get("num_queries", 10000))
    rate_qps = arrival.get("rate_qps")
    lam = float(rate_qps) if arrival.get("mode") == "poisson" and rate_qps else 0.0
    raw_dump = bool(arrival.get("raw_dump", False))

    out_file = ""
    if raw_dump:
        results_dir = Path("/results")
        if results_dir.exists():
            out_file = str(results_dir / "openloop_raw.txt")

    cmd = [
        _tool("search_openloop"),
        element_type,
        str(prefix_path),
        str(num_workers),
        str(beam_width),
        str(queries_bin),
        gt_arg,
        str(k),
        metric,
        nbr_type,
        str(mode),
        str(mem_L),
        str(Ls),
        str(lam),
        str(num_queries),
    ]
    if out_file:
        cmd.append(out_file)

    proc = _run(cmd, env=env)
    values: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("RESULT"):
            continue
        values = {mtch.group(1): float(mtch.group(2)) for mtch in _OPENLOOP_KV_RE.finditer(line)}
        break

    if not values:
        raise RuntimeError(
            "search_openloop produced no parseable RESULT line; check stdout for errors"
        )

    achieved_qps = values.get("achieved_qps", 0.0)
    lat_mean_ms = values.get("lat_mean", 0.0) / 1000.0
    total_time_seconds = num_queries / achieved_qps if achieved_qps > 0 else 0.0

    open_loop = {
        "achieved_qps": achieved_qps,
        "arrival_rate_qps": rate_qps if lam > 0 else None,
        "num_queries": num_queries,
        "num_workers": num_workers,
        "warmup_fraction": 0.1,  # fixed by the driver (drops first 10%)
        "latency_ms": {
            "mean": lat_mean_ms,
            "p50": values.get("p50", 0.0) / 1000.0,
            "p90": values.get("p90", 0.0) / 1000.0,
            "p99": values.get("p99", 0.0) / 1000.0,
            "p999": values.get("p999", 0.0) / 1000.0,
        },
        "service_time_ms": {
            "mean": values.get("svc_mean", 0.0) / 1000.0,
            "p99": values.get("svc_p99", 0.0) / 1000.0,
        },
        "ios_per_query": values.get("ios"),
        "device_read_iops": values.get("dev_iops"),
        "device_avg_read_latency_ms": values.get("dev_lat"),
        "device_util": values.get("dev_util"),
        "machine_cpu_util": values.get("cpu"),
        "raw_dump_path": out_file or None,
    }

    return {
        "total_queries": num_queries,
        "total_time_seconds": total_time_seconds,
        "qps": achieved_qps,
        "recall": values.get("recall"),
        "mean_latency_ms": lat_mean_ms,
        "p50_latency_ms": open_loop["latency_ms"]["p50"],
        "p95_latency_ms": None,
        "p99_latency_ms": open_loop["latency_ms"]["p99"],
        "max_latency_ms": None,
        "stats": (
            {"io_reads": int(round(values["ios"] * num_queries))} if "ios" in values else None
        ),
        "open_loop": open_loop,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PipeANN Algorithm Runner")
    parser.add_argument("--mode", choices=["build", "search"], required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as exc:
        print(json.dumps({"status": "error", "error_message": str(exc)}))
        raise SystemExit(1) from exc

    result = run_build(config) if args.mode == "build" else run_search(config)
    print(json.dumps(result))

    results_dir = Path("/results")
    if results_dir.exists():
        (results_dir / "metrics.json").write_text(json.dumps(result))

    raise SystemExit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
