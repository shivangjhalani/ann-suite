"""ANN Suite runner for Starling (https://github.com/zilliztech/starling).

Build pipeline (mirrors upstream scripts/run_benchmark.sh and the isfcr
research host's /home/isfcr/starling/benchmark/benchmark.py):
  1. build_disk_index                      -> <p>_disk.index (+ PQ files)
  2. (optional) gen_random_slice + build_memory_index -> in-memory nav graph
  3. graph_partition/partitioner on the Vamana index -> gp/part.bin
  4. index_relayout                         -> page-relayout disk index
  5. install the relayout as <p>_disk.index and gp/part.bin as <p>_partition.bin

Search runs search_disk_index with page search on. Starling keeps original
vector IDs (the partition file maps IDs to pages), so recall comes straight
from the binary. "Mean IOs" counts one 4 KB O_DIRECT sector read per IO and is
reported as stats.io_reads (totals over the run).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

METRIC_MAP = {"L2": "l2", "euclidean": "l2", "IP": "mips", "inner_product": "mips"}
PREFIX = "starling"


def _now() -> str:
    # The image runs Python 3.10 (ubuntu:jammy), which has no datetime.UTC.
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def _bin(rel: str) -> str:
    return str(Path(os.environ.get("STARLING_BUILD", "/opt/starling/release")) / rel)


def _type_for(dtype: np.dtype) -> str:
    if dtype == np.uint8:
        return "uint8"
    if dtype == np.int8:
        return "int8"
    return "float"


def _write_bin(path: Path, data: np.ndarray) -> str:
    element_type = _type_for(data.dtype)
    out_dtype = {"uint8": np.uint8, "int8": np.int8, "float": np.float32}[element_type]
    with path.open("wb") as f:
        np.asarray(data.shape[:2], dtype=np.uint32).tofile(f)
        for i in range(0, data.shape[0], 1_000_000):
            np.ascontiguousarray(data[i : i + 1_000_000], dtype=out_dtype).tofile(f)
    return element_type


def _write_gt(path: Path, ids: np.ndarray) -> None:
    ids = np.ascontiguousarray(ids, dtype=np.uint32)
    with path.open("wb") as f:
        np.asarray(ids.shape, dtype=np.int32).tofile(f)
        ids.tofile(f)


def _run(cmd: list[str], log: Path) -> str:
    print("$ " + " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    if proc.returncode:
        print(proc.stdout[-4000:], proc.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd[0]}")
    return proc.stdout


def run_build(config: dict[str, Any]) -> dict[str, Any]:
    try:
        start = time.perf_counter()
        index_path = Path(config["index_path"])
        index_path.mkdir(parents=True, exist_ok=True)
        log = index_path / "build.log"
        data = np.load(config["dataset_path"], mmap_mode="r")
        metric = METRIC_MAP[config.get("metric", "L2")]
        args = dict(config.get("build_args", {}))
        T = int(args.get("num_threads", os.cpu_count() or 8))
        prefix = index_path / PREFIX
        data_bin = index_path / "data.bin"
        dtype = _write_bin(data_bin, data)

        _run(
            [
                _bin("tests/build_disk_index"),
                "--data_type",
                dtype,
                "--dist_fn",
                metric,
                "--data_path",
                str(data_bin),
                "--index_path_prefix",
                str(prefix),
                "-R",
                str(int(args.get("R", 48))),
                "-L",
                str(int(args.get("L", 128))),
                "-B",
                str(float(args.get("B", 0.1))),
                "-M",
                str(float(args.get("M", 32))),
                "-T",
                str(T),
            ],
            log,
        )

        mem_built = False
        if args.get("build_mem_index", True):
            sample = index_path / "sample"
            _run(
                [
                    _bin("tests/utils/gen_random_slice"),
                    dtype,
                    str(data_bin),
                    str(sample),
                    str(float(args.get("mem_sample_rate", 0.01))),
                ],
                log,
            )
            _run(
                [
                    _bin("tests/build_memory_index"),
                    "--data_type",
                    dtype,
                    "--dist_fn",
                    metric,
                    "--data_path",  # a prefix: the tool reads <p>_data.bin and <p>_ids.bin
                    str(sample),
                    "--index_path_prefix",
                    str(index_path / "mem_index"),
                    "-R",
                    str(int(args.get("mem_R", 48))),
                    "-L",
                    str(int(args.get("mem_L", 128))),
                    "--alpha",
                    "1.2",
                    "-T",
                    str(T),
                ],
                log,
            )
            mem_built = True

        # Partition + relayout read the plain Vamana index; keep it as the
        # "beam" copy only for the duration of the build.
        disk = Path(f"{prefix}_disk.index")
        beam = Path(f"{prefix}_disk_beam_search.index")
        shutil.move(disk, beam)
        gp_dir = index_path / "gp"
        gp_dir.mkdir(exist_ok=True)
        gp_file = gp_dir / "part.bin"
        _run(
            [
                _bin("graph_partition/partitioner"),
                "--index_file",
                str(beam),
                "--data_type",
                dtype,
                "--gp_file",
                str(gp_file),
                "-T",
                str(T),
                "--ldg_times",
                str(int(args.get("ldg_times", 16))),
            ],
            log,
        )
        _run([_bin("tests/utils/index_relayout"), str(beam), str(gp_file)], log)
        shutil.move(gp_dir / "part_tmp.index", disk)
        shutil.copy2(gp_file, Path(f"{prefix}_partition.bin"))
        beam.unlink()
        data_bin.unlink(missing_ok=True)

        manifest = {"data_type": dtype, "metric": metric, "mem_index": mem_built, **args}
        (index_path / "starling_manifest.json").write_text(json.dumps(manifest, indent=2))
        index_size = sum(p.stat().st_size for p in index_path.rglob("*") if p.is_file())
        return {
            "status": "success",
            "build_time_seconds": time.perf_counter() - start,
            "index_size_bytes": index_size,
            "mem_index_built": mem_built,
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


def _parse_row(stdout: str, L: int) -> dict[str, float]:
    """Upstream row: L W QPS MeanLat P999Lat MeanIOs CPU(s) B4Load AfterCache PeakMem Recall."""
    for line in stdout.splitlines():
        toks = line.split()
        if len(toks) < 11 or toks[0] != str(L):
            continue
        try:
            v = [float(t) for t in toks]
        except ValueError:
            continue
        recall = v[10]
        return {
            "qps": v[2],
            "mean_latency_us": v[3],
            "p999_latency_us": v[4],
            "mean_ios": v[5],
            "recall": recall / 100.0 if recall > 1.0 else recall,
        }
    raise RuntimeError(f"no result row for L={L} in search_disk_index output")


def run_search(config: dict[str, Any]) -> dict[str, Any]:
    try:
        index_path = Path(config["index_path"])
        manifest = json.loads((index_path / "starling_manifest.json").read_text())
        k = int(config.get("k", 10))
        args = dict(config.get("search_args", {}))
        L = int(args.get("Ls", 100))
        work = Path("/tmp/starling_search")
        work.mkdir(parents=True, exist_ok=True)
        queries = np.load(config["queries_path"])
        qbin = work / "queries.bin"
        _write_bin(qbin, queries)
        gt_arg = "null"
        if config.get("ground_truth_path"):
            _write_gt(work / "gt.bin", np.load(config["ground_truth_path"])[:, :k])
            gt_arg = str(work / "gt.bin")
        prefix = index_path / PREFIX
        mem_L = int(args.get("mem_L", 0))
        cmd = [
            _bin("tests/search_disk_index"),
            "--data_type",
            manifest["data_type"],
            "--dist_fn",
            manifest["metric"],
            "--index_path_prefix",
            str(prefix),
            "--query_file",
            str(qbin),
            "--gt_file",
            gt_arg,
            "-K",
            str(k),
            "--result_path",
            str(work / "res"),
            "--use_page_search",
            "1",
            "--use_sq",
            "0",
            "--disk_file_path",
            f"{prefix}_disk.index",
            "-T",
            str(int(args.get("num_threads", 8))),
            "-W",
            str(int(args.get("beam_width", 4))),
            "--num_nodes_to_cache",
            str(int(args.get("num_nodes_to_cache", 0))),
            "--use_ratio",
            str(float(args.get("use_ratio", 1.0))),
            "--mem_L",
            str(mem_L),
        ]
        if mem_L > 0:
            if not manifest.get("mem_index"):
                raise ValueError("mem_L > 0 but this index was built without a memory index")
            cmd += ["--mem_index_path", str(index_path / "mem_index")]
        cmd += ["-L", str(L)]

        q_start = _now()
        t0 = time.perf_counter()
        out = _run(cmd, work / "search.log")
        wall = time.perf_counter() - t0
        q_end = _now()
        row = _parse_row(out, L)
        nq = len(queries)
        lat_ms = row["mean_latency_us"] / 1000.0
        return {
            "status": "success",
            "total_queries": nq,
            "total_time_seconds": nq / row["qps"] if row["qps"] else wall,
            "qps": row["qps"],
            "recall": row["recall"],
            "mean_latency_ms": lat_ms,
            "p50_latency_ms": lat_ms,
            "p95_latency_ms": lat_ms,
            "p99_latency_ms": row["p999_latency_us"] / 1000.0,
            "warmup_duration_seconds": 0.0,
            "warmup_start_timestamp": q_start,
            "warmup_end_timestamp": q_start,
            "load_duration_seconds": 0.0,
            "query_start_timestamp": q_start,
            "query_end_timestamp": q_end,
            "process_wall_seconds": wall,
            "stats": {"io_reads": int(round(row["mean_ios"] * nq))},
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
    parser = argparse.ArgumentParser(description="Starling Algorithm Runner")
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
