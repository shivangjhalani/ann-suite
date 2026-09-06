"""ANN Suite runner for Microsoft's SPANN implementation in SPTAG.

SPTAG's command-line tools use a binary format with a two-int32 header followed
by row-major vectors. The runner converts the suite's NumPy files, builds the
SPANN index on the mounted index volume, and uses IndexSearcher for queries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from utils import compute_recall


def _binary_path(path: Path, data: np.ndarray) -> Path:
    """Write data in SPTAG DEFAULT format and return its path."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    with path.open("wb") as output:
        np.asarray([len(data), data.shape[1]], dtype=np.int32).tofile(output)
        data.tofile(output)
    return path


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run an SPTAG command and preserve its diagnostics on stderr."""
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, file=sys.stderr, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode:
        raise RuntimeError(f"SPTAG command failed ({result.returncode}): {' '.join(command)}")
    return result


def _tool(name: str) -> str:
    # SPTAG's CMake targets are lowercase on Linux (indexbuilder/indexsearcher).
    return str(Path(os.environ.get("SPTAG_BIN", "/opt/sptag/Release")) / name.lower())


def _write_config(
    index_path: Path, base_path: Path, dimension: int, metric: str, args: dict[str, Any]
) -> Path:
    """Create the SPTAG INI consumed by IndexBuilder and IndexSearcher."""
    dist = "Cosine" if metric.lower() in {"cosine", "angular"} else "L2"
    threads = int(args.get("num_threads", 4))
    config = f"""[Base]
ValueType=Float
DistCalcMethod={dist}
IndexAlgoType=BKT
Dim={dimension}
VectorPath={base_path}
VectorType=DEFAULT
IndexDirectory={index_path}

[SelectHead]
isExecute=true
TreeNumber=1
BKTKmeansK={int(args.get("kmeans_k", 32))}
BKTLeafSize={int(args.get("leaf_size", 8))}
SamplesNumber={int(args.get("samples", 1000))}
SelectThreshold={int(args.get("select_threshold", 10))}
SplitFactor={int(args.get("split_factor", 6))}
SplitThreshold={int(args.get("split_threshold", 25))}
Ratio={float(args.get("ratio", 0.12))}
NumberOfThreads={threads}

[BuildHead]
isExecute=true
NeighborhoodSize={int(args.get("neighborhood_size", 32))}
TPTNumber={int(args.get("tpt_number", 32))}
TPTLeafSize={int(args.get("tpt_leaf_size", 2000))}
MaxCheck={int(args.get("max_check", 16324))}
MaxCheckForRefineGraph={int(args.get("max_check", 16324))}
RefineIterations={int(args.get("refine_iterations", 3))}
NumberOfThreads={threads}

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum={int(args.get("internal_result_num", 64))}
ReplicaCount={int(args.get("replica_count", 8))}
PostingPageLimit={int(args.get("posting_page_limit", 3))}
NumberOfThreads={threads}
MaxCheck={int(args.get("max_check", 16324))}
TmpDir={index_path}
"""
    config_path = index_path / "spann.ini"
    config_path.write_text(config)
    return config_path


def run_build(config: dict[str, Any]) -> dict[str, Any]:
    try:
        data = np.load(Path(config["dataset_path"])).astype(np.float32)
        index_path = Path(config["index_path"])
        index_path.mkdir(parents=True, exist_ok=True)
        base_path = _binary_path(index_path / "base.bin", data)
        args = dict(config.get("build_args", {}))
        ini = _write_config(index_path, base_path, data.shape[1], config.get("metric", "L2"), args)
        start = time.perf_counter()
        _run(
            [
                _tool("indexbuilder"),
                "-c",
                str(ini),
                "-d",
                str(data.shape[1]),
                "-v",
                "Float",
                "-f",
                "DEFAULT",
                "-o",
                str(index_path),
                "-a",
                "SPANN",
            ]
        )
        build_time = time.perf_counter() - start
        index_size = sum(path.stat().st_size for path in index_path.rglob("*") if path.is_file())
        return {
            "status": "success",
            "build_time_seconds": build_time,
            "index_size_bytes": index_size,
        }
    except Exception as exc:
        print(f"SPANN build failed: {exc}", file=sys.stderr)
        return {
            "status": "error",
            "error_message": str(exc),
            "build_time_seconds": 0,
            "index_size_bytes": 0,
        }


def _parse_results(path: Path, query_count: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.full((query_count, k), -1, dtype=np.int64)
    distances = np.full((query_count, k), np.inf, dtype=np.float32)
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        query_id, values = line.split(":", 1)
        row = int(query_id)
        for col, value in enumerate(values.split("|")[:k]):
            if "@" not in value:
                continue
            distance, identifier = value.split("@", 1)
            if identifier != "NULL":
                distances[row, col] = float(distance)
                indices[row, col] = int(identifier)
    return indices, distances


def run_search(config: dict[str, Any]) -> dict[str, Any]:
    try:
        index_path = Path(config["index_path"])
        queries = np.load(Path(config["queries_path"])).astype(np.float32)
        k = int(config.get("k", 10))
        query_rounds = int(config.get("query_rounds", 1))
        search_args = dict(config.get("search_args", {}))
        result_path = index_path / "search-results.txt"
        queries_bin = _binary_path(index_path / "queries.bin", queries)
        command = [
            _tool("indexsearcher"),
            "-i",
            str(queries_bin),
            "-x",
            str(index_path),
            "-o",
            str(result_path),
            "-d",
            str(queries.shape[1]),
            "-v",
            "Float",
            "-f",
            "DEFAULT",
            "-k",
            str(k),
            "-b",
            str(len(queries)),
            "-of",
            "0",
            "BuildSSDIndex.SearchInternalResultNum="
            + str(search_args.get("internal_result_num", 64)),
            "BuildSSDIndex.SearchPostingPageLimit=" + str(search_args.get("posting_page_limit", 3)),
        ]
        num_threads = min(int(search_args.get("num_threads", 8)), 16)
        command.extend(["-t", str(max(1, num_threads))])
        ground_truth = None
        if config.get("ground_truth_path"):
            ground_truth = np.load(Path(config["ground_truth_path"]))
        total_queries = 0
        total_search_time = 0.0
        load_duration = 0.0
        indices = None
        query_start = datetime.now(UTC).isoformat()
        first_load_start = time.perf_counter()
        for round_idx in range(query_rounds):
            result_path.unlink(missing_ok=True)
            round_start = time.perf_counter()
            _run(command, cwd=index_path)
            round_elapsed = time.perf_counter() - round_start
            total_queries += len(queries)
            total_search_time += round_elapsed
            if round_idx == 0:
                load_duration = time.perf_counter() - first_load_start
            indices, _ = _parse_results(result_path, len(queries), k)
        query_end = datetime.now(UTC).isoformat()
        if indices is None or indices.size == 0:
            raise RuntimeError("SPTAG returned no search results")
        return {
            "status": "success",
            "total_queries": total_queries,
            "total_time_seconds": total_search_time,
            "qps": total_queries / total_search_time,
            "recall": compute_recall(indices, ground_truth, k)
            if ground_truth is not None
            else None,
            "mean_latency_ms": total_search_time * 1000 / total_queries,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "p99_latency_ms": None,
            "max_latency_ms": None,
            "warmup_duration_seconds": 0.0,
            "query_start_timestamp": query_start,
            "query_end_timestamp": query_end,
            "load_duration_seconds": load_duration,
            "cache_warmup_queries_requested": 0,
            "cache_warmup_queries_executed": 0,
            "cache_warmup_duration_seconds": 0.0,
        }
    except Exception as exc:
        print(f"SPANN search failed: {exc}", file=sys.stderr)
        return {
            "status": "error",
            "error_message": str(exc),
            "total_queries": 0,
            "total_time_seconds": 0,
            "qps": 0,
            "recall": None,
            "mean_latency_ms": 0.0,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "p99_latency_ms": None,
            "max_latency_ms": None,
            "warmup_duration_seconds": 0.0,
            "query_start_timestamp": None,
            "query_end_timestamp": None,
            "load_duration_seconds": 0.0,
            "cache_warmup_queries_requested": 0,
            "cache_warmup_queries_executed": 0,
            "cache_warmup_duration_seconds": 0.0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="SPANN Algorithm Runner")
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
    raise SystemExit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
