"""ANN Suite runner for Microsoft's SPANN implementation in SPTAG.

SPTAG's command-line tools use a binary format with a two-int32 header followed
by row-major vectors. The runner converts the suite's NumPy files, builds the
SPANN index on the mounted index volume, and uses IndexSearcher for queries.

Search metrics are SPTAG's own in-process measurements: the runner timestamps
indexsearcher's output lines to separate index load (reported as warmup) from the
timed query loop, so QPS, latency and the resource-metrics window exclude loading.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from utils import compute_recall

# SPTAG ValueType name -> NumPy dtype for the query file.
VALUE_TYPES: dict[str, type[np.generic]] = {
    "Float": np.float32,
    "UInt8": np.uint8,
    "Int8": np.int8,
    "Int16": np.int16,
}


def _binary_path(path: Path, data: np.ndarray, dtype: type[np.generic] = np.float32) -> Path:
    """Write data in SPTAG DEFAULT format and return its path."""
    data = np.ascontiguousarray(data, dtype=dtype)
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


MAX_EMPTY_ROW_FRACTION = 0.01

# SPTAG's default [BuildSSDIndex] NumberOfThreads (m_iSSDNumberOfThreads), used when the
# persisted indexloader.ini does not set it.
SPTAG_DEFAULT_SSD_THREADS = 16

# indexsearcher prints this header immediately before the timed query loop starts.
SEARCH_HEADER = "[query]"
# ...and one line per batch right after the batch's threads join:
# "<start>-<end>\t<maxcheck>\t<avg s>\t<p99 s>\t<p95 s>\t<recall>\t\t<qps>\t\t<mem>GB".
# The run summary line has the same prefix but no "GB" column, so it does not match.
BATCH_LINE = re.compile(
    r"(\d+)-(\d+)\t\S+\t([\d.]+)\t([\d.]+)\t([\d.]+)\t[\d.]+\t\t([\d.]+)\t\t\d+GB"
)


def check_results(indices: np.ndarray) -> None:
    """Fail on empty result rows. SPTAG's indexsearcher exits 0 with empty results when it
    cannot allocate (e.g. under a container/cgroup memory limit), which would otherwise be
    reported as a fast, recall-0 "success"."""
    empty = float((indices < 0).all(axis=1).mean())
    if empty > MAX_EMPTY_ROW_FRACTION:
        raise RuntimeError(
            f"SPTAG returned no neighbours for {empty:.0%} of queries; likely out of memory "
            "under the configured memory limit"
        )


def _read_loader_ini(index_path: Path) -> configparser.ConfigParser:
    # strict=False: SPANN's SaveConfig writes BuildHead and head-index keys into one
    # section, so a key can repeat.
    ini = configparser.ConfigParser(interpolation=None, strict=False)
    ini.optionxform = str  # type: ignore[assignment,method-assign]
    ini.read(index_path / "indexloader.ini")
    return ini


def search_thread_cap(index_path: Path) -> int:
    """Highest safe IndexSearcher thread count for a persisted SPANN index.

    The SSD searcher sizes its workspace pool from [BuildSSDIndex] NumberOfThreads as
    persisted in indexloader.ini at load time; searching with more threads segfaults
    (FreeWorkSpaceIds is not initialized). The -t flag is applied only after loading.
    """
    ini = _read_loader_ini(index_path)
    return ini.getint("BuildSSDIndex", "NumberOfThreads", fallback=SPTAG_DEFAULT_SSD_THREADS)


def check_head_parameters(index_path: Path) -> None:
    """Refuse an indexloader.ini without the head index's parameters.

    SPTAG's own SaveConfig writes the head (BKT) index parameters under [BuildHead]. A
    hand-written indexloader.ini that omits them still loads, but the head then uses BKT
    defaults, e.g. DistCalcMethod=Cosine on an L2 index: recall drops with unchanged IO.
    """
    ini = _read_loader_ini(index_path)
    if not ini.has_option("BuildHead", "DistCalcMethod"):
        raise ValueError(
            f"{index_path / 'indexloader.ini'} has no head-index parameters under "
            "[BuildHead] (e.g. DistCalcMethod); SPTAG would load the head with BKT "
            "defaults (Cosine distance). Copy head_index/indexloader.ini's [Index] "
            "parameters into [BuildHead]."
        )


@dataclass
class SearchRound:
    """One indexsearcher process: its load phase and its in-process timed search."""

    process_start: datetime
    search_start: datetime
    search_end: datetime
    queries: int
    search_seconds: float  # sum of SPTAG's per-batch wall times (excludes load/output)
    mean_latency_s: float
    max_latency_s: float
    p95_latency_s: float
    p99_latency_s: float


def parse_round(
    process_start: datetime, lines: list[tuple[datetime, str]], latency_log: str
) -> SearchRound:
    """Build a SearchRound from timestamped indexsearcher output and Recall-result.out.

    Recall-result.out holds "mean std min max " (seconds, full precision) per batch. The
    p95/p99 come from the batch lines, which SPTAG prints with %.4f seconds, so they have
    0.1 ms resolution. With several batches they are query-weighted means of per-batch
    percentiles (exact for the runner's single batch).
    """
    search_start = next((t for t, line in lines if SEARCH_HEADER in line), None)
    batches = [(t, m) for t, line in lines if (m := BATCH_LINE.search(line))]
    if search_start is None or not batches:
        raise RuntimeError("could not find the search phase in indexsearcher output")
    stats = [float(v) for v in latency_log.split()]
    if len(stats) != 4 * len(batches):
        raise RuntimeError(
            f"Recall-result.out has {len(stats)} values for {len(batches)} batches (want 4 each)"
        )
    sizes = [int(m.group(2)) - int(m.group(1)) for _, m in batches]
    total = sum(sizes)

    def weighted(values: list[float]) -> float:
        return sum(n * v for n, v in zip(sizes, values, strict=True)) / total

    return SearchRound(
        process_start=process_start,
        search_start=search_start,
        search_end=batches[-1][0],
        queries=total,
        search_seconds=sum(n / float(m.group(6)) for n, (_, m) in zip(sizes, batches, strict=True)),
        mean_latency_s=weighted(stats[0::4]),
        max_latency_s=max(stats[3::4]),
        p95_latency_s=weighted([float(m.group(5)) for _, m in batches]),
        p99_latency_s=weighted([float(m.group(4)) for _, m in batches]),
    )


def _run_search_round(command: list[str], cwd: Path) -> SearchRound:
    """Run indexsearcher, timestamping each output line as it arrives (SPTAG flushes
    stdout after every log line), so the load and search phases can be told apart."""
    latency_log = cwd / "Recall-result.out"  # indexsearcher appends to it in its cwd
    latency_log.unlink(missing_ok=True)
    process_start = datetime.now(UTC)
    proc = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    lines: list[tuple[datetime, str]] = []
    for line in proc.stdout:
        lines.append((datetime.now(UTC), line))
        print(line, file=sys.stderr, end="")
    if proc.wait():
        raise RuntimeError(f"SPTAG command failed ({proc.returncode}): {' '.join(command)}")
    return parse_round(process_start, lines, latency_log.read_text())


def run_search(config: dict[str, Any]) -> dict[str, Any]:
    try:
        index_path = Path(config["index_path"])
        queries = np.load(Path(config["queries_path"])).astype(np.float32)
        k = int(config.get("k", 10))
        query_rounds = int(config.get("query_rounds", 1))
        search_args = dict(config.get("search_args", {}))
        result_path = index_path / "search-results.txt"
        check_head_parameters(index_path)
        # Must match the index's ValueType (e.g. UInt8 for BIGANN indexes built natively).
        value_type = str(search_args.get("value_type", "Float"))
        dtype = VALUE_TYPES[value_type]
        if not np.array_equal(queries.astype(dtype).astype(np.float32), queries):
            raise ValueError(f"queries are not exactly representable as {value_type}")
        queries_bin = _binary_path(index_path / "queries.bin", queries, dtype)
        requested_threads = int(search_args.get("num_threads", 8))
        thread_cap = search_thread_cap(index_path)
        num_threads = max(1, min(requested_threads, thread_cap))
        if num_threads != requested_threads:
            print(
                f"WARNING: num_threads={requested_threads} capped to {num_threads}: the index "
                f"was persisted with [BuildSSDIndex] NumberOfThreads={thread_cap}",
                file=sys.stderr,
            )
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
            value_type,
            "-f",
            "DEFAULT",
            "-k",
            str(k),
            "-b",
            str(len(queries)),
            "-of",
            "0",
            "-t",
            str(num_threads),
            "BuildSSDIndex.SearchInternalResultNum="
            + str(search_args.get("internal_result_num", 64)),
            "BuildSSDIndex.SearchPostingPageLimit=" + str(search_args.get("posting_page_limit", 3)),
        ]
        ground_truth = None
        if config.get("ground_truth_path"):
            ground_truth = np.load(Path(config["ground_truth_path"]))
        rounds: list[SearchRound] = []
        indices = None
        for _ in range(query_rounds):
            result_path.unlink(missing_ok=True)
            rounds.append(_run_search_round(command, index_path))
            indices, _ = _parse_results(result_path, len(queries), k)
        if indices is None or indices.size == 0:
            raise RuntimeError("SPTAG returned no search results")
        check_results(indices)
        if query_rounds > 1:
            print(
                "WARNING: query_rounds > 1 runs one indexsearcher process per round; the "
                "resource-metrics window spans all rounds, including later rounds' index load",
                file=sys.stderr,
            )
        total_queries = sum(r.queries for r in rounds)
        search_seconds = sum(r.search_seconds for r in rounds)
        first = rounds[0]
        load_seconds = (first.search_start - first.process_start).total_seconds()

        def weighted_ms(field: str) -> float:
            return 1000 * sum(getattr(r, field) * r.queries for r in rounds) / total_queries

        return {
            "status": "success",
            "total_queries": total_queries,
            # In-process search time as SPTAG measures it: excludes index load and
            # result-file output, which a process wall clock would include.
            "total_time_seconds": search_seconds,
            "qps": total_queries / search_seconds,
            "recall": compute_recall(indices, ground_truth, k)
            if ground_truth is not None
            else None,
            "mean_latency_ms": weighted_ms("mean_latency_s"),
            "p50_latency_ms": None,  # indexsearcher does not report a median
            "p95_latency_ms": weighted_ms("p95_latency_s"),
            "p99_latency_ms": weighted_ms("p99_latency_s"),
            "max_latency_ms": 1000 * max(r.max_latency_s for r in rounds),
            # Index load of the first round = the suite's "warmup" phase.
            "warmup_duration_seconds": load_seconds,
            "warmup_start_timestamp": first.process_start.isoformat(),
            "warmup_end_timestamp": first.search_start.isoformat(),
            "query_start_timestamp": first.search_start.isoformat(),
            "query_end_timestamp": rounds[-1].search_end.isoformat(),
            "load_duration_seconds": load_seconds,
            "cache_warmup_queries_requested": 0,
            "cache_warmup_queries_executed": 0,
            "cache_warmup_duration_seconds": 0.0,
            "stats": {"search_threads": num_threads},
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
