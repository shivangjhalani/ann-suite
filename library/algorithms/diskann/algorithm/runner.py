"""DiskANN Algorithm Runner using diskannpy.

This module implements the ANN benchmarking suite protocol for DiskANN.
Based on big-ann-benchmarks implementation pattern.

DiskANN is a disk-based graph algorithm that provides efficient billion-scale
search with minimal memory footprint by storing the graph on SSD.

Build Parameters:
    R: Graph degree (number of neighbors per node)
    L: Build-time search list size
    num_threads: Number of threads for parallel operations
    pq_disk_bytes: Bytes for PQ compression on disk (0 = uncompressed, full accuracy)
    build_memory_maximum: ADVISORY hint for index partitioning (in GB). This is NOT
        a hard memory limit - actual build memory usage will exceed this value.
        It tells DiskANN how to optimize the index structure for target search RAM.
    search_memory_maximum: ADVISORY hint for search-time memory budget (in GB).
        Controls how the index is laid out for disk access patterns.

Search Parameters:
    Ls: Search list size (higher = better recall, slower)
    beam_width: Beam width for graph traversal
    num_threads: Number of threads for search
    num_nodes_to_cache: Number of nodes to cache in RAM (default 0 for minimum memory)

CRITICAL: This is a disk-based algorithm. All index files are written to
/data/index/ to ensure accurate I/O metrics from the host monitor.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import diskannpy
import numpy as np
from utils import compute_recall


class DiskANNIndex:
    """DiskANN index wrapper using diskannpy."""

    METRIC_MAP = {
        "L2": "l2",
        "euclidean": "l2",
        "IP": "mips",
        "inner_product": "mips",
        "cosine": "cosine",
        "angular": "cosine",
    }

    def __init__(
        self,
        R: int = 64,
        L: int = 100,
        num_threads: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize DiskANN index.

        Args:
            R: Graph degree
            L: Build-time search list size
            num_threads: Number of threads
        """
        self.R = R
        self.L = L
        self.num_threads = num_threads
        self.index = None
        self.dimension: int = 0
        self.metric: str = "l2"
        self.index_path: Path | None = None

    def build(
        self,
        data: np.ndarray,
        index_path: Path,
        metric: str = "L2",
        pq_disk_bytes: int = 0,
        build_memory_maximum: float = 2.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the DiskANN index.

        Args:
            data: Base vectors (N x D)
            index_path: Where to save the index
            metric: Distance metric
            pq_disk_bytes: PQ bytes for disk storage (0 = uncompressed)
            build_memory_maximum: ADVISORY hint for index partitioning (GB).
                This is NOT a hard memory limit. Actual build memory will be
                significantly higher as it needs to load all vectors and
                construct the graph. This parameter tells diskannpy how to
                optimize the index layout for the target search memory budget.

        Returns:
            Build statistics
        """
        n_vectors, dimension = data.shape
        self.dimension = dimension
        if metric not in self.METRIC_MAP:
            raise ValueError(f"Unsupported DiskANN metric: {metric}")
        self.metric = self.METRIC_MAP[metric]
        self.index_path = Path(index_path)

        # Create index directory (clean if exists to avoid diskannpy errors)
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Save data in DiskANN format (needs to be on disk for build)
        data_file = self.index_path / "data.bin"

        # Write vectors in diskann binary format
        # Format: [uint32 num_points][uint32 dimension][float32 vectors...]
        with open(data_file, "wb") as f:
            np.array([n_vectors], dtype=np.uint32).tofile(f)
            np.array([dimension], dtype=np.uint32).tofile(f)
            data.astype(np.float32).tofile(f)

        # Build index
        start_time = time.perf_counter()

        # Use diskannpy's build_disk_index function
        diskannpy.build_disk_index(
            data=str(data_file),
            distance_metric=self.metric,
            index_directory=str(self.index_path),
            complexity=self.L,
            graph_degree=self.R,
            num_threads=self.num_threads,
            pq_disk_bytes=pq_disk_bytes,
            build_memory_maximum=build_memory_maximum,
            search_memory_maximum=kwargs.get("search_memory_maximum", 0.5),
            vector_dtype=np.float32,
        )

        build_time = time.perf_counter() - start_time

        # Calculate index size (sum of all index files)
        index_size = sum(f.stat().st_size for f in self.index_path.rglob("*") if f.is_file())

        return {
            "build_time_seconds": build_time,
            "index_size_bytes": index_size,
        }

    def load(
        self,
        index_path: Path,
        dimension: int,
        metric: str = "L2",
        num_nodes_to_cache: int = 0,
        index_prefix: str = "ann",
        vector_dtype: str | None = None,
    ) -> None:
        """Load a previously built index.

        Args:
            index_path: Path to the index directory
            dimension: Vector dimension
            metric: Distance metric
            num_nodes_to_cache: Number of nodes to cache in RAM
            index_prefix: File prefix for the index components (default "ann",
                matching diskannpy's convention: ann_disk.index, etc.). Pass the
                prefix used when the index was built.
            vector_dtype: Dtype of the index vectors ("float32" or "uint8").
                Indexes built from uint8 (bvecs) data must be searched with
                uint8, not float32. Defaults to float32.
        """
        self.dimension = dimension
        if metric not in self.METRIC_MAP:
            raise ValueError(f"Unsupported DiskANN metric: {metric}")
        self.metric = self.METRIC_MAP[metric]
        self.index_path = Path(index_path)
        if vector_dtype is None:
            self.vector_dtype: np.dtype = np.dtype(np.float32)
        elif str(vector_dtype) == "uint8":
            self.vector_dtype = np.dtype(np.uint8)
        else:
            self.vector_dtype = np.dtype(np.float32)

        # Load the index with required parameters for diskannpy
        # Use StaticDiskIndex for true disk-based search
        self.index = diskannpy.StaticDiskIndex(
            index_directory=str(self.index_path),
            num_threads=self.num_threads,
            num_nodes_to_cache=num_nodes_to_cache,
            distance_metric=self.metric,
            vector_dtype=self.vector_dtype,
            dimensions=dimension,
            index_prefix=index_prefix,
        )

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        Ls: int = 100,
        beam_width: int = 2,
        batch_mode: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Search for nearest neighbors.

        Args:
            queries: Query vectors (Q x D)
            k: Number of neighbors to return
            Ls: Search list size (complexity)
            beam_width: Beam width for search

        Returns:
            Tuple of (indices, distances, latencies_ms)
        """
        if self.index is None:
            raise RuntimeError("Index not loaded")

        latencies: list[float] = []
        all_indices: list[np.ndarray] = []
        all_distances: list[np.ndarray] = []

        if batch_mode:
            # Batch Mode: Highest Throughput
            start = time.perf_counter()

            # Try batch_search if available (diskannpy.StaticDiskIndex)
            # API: batch_search(queries, k_neighbors, complexity, num_threads, beam_width=2)
            if hasattr(self.index, "batch_search"):
                all_indices, all_distances = self.index.batch_search(
                    queries, k, Ls, self.num_threads, beam_width
                )
                total_elapsed_ms = (time.perf_counter() - start) * 1000
                mean_latency = total_elapsed_ms / len(queries)
                latencies = [mean_latency] * len(queries)
            else:
                # Fallback to serial search if batch API unavailable
                for query in queries:
                    s = time.perf_counter()
                    labels, distances = self.index.search(
                        query,
                        k_neighbors=k,
                        complexity=Ls,
                        beam_width=beam_width,
                    )
                    latencies.append((time.perf_counter() - s) * 1000)
                    all_indices.append(labels)
                    all_distances.append(distances)
        else:
            # Serial Mode: Accurate Latency Distribution
            for query in queries:
                start = time.perf_counter()
                labels, distances = self.index.search(
                    query,
                    k_neighbors=k,
                    complexity=Ls,
                    beam_width=beam_width,
                )
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
                all_indices.append(labels)
                all_distances.append(distances)

        return np.asarray(all_indices), np.asarray(all_distances), latencies

    def _native_index(self) -> Any:
        """Return the underlying native (pybind11) index object.

        ``diskannpy.StaticDiskIndex`` is a pure-Python wrapper that delegates to a
        native ``_index`` attribute (e.g. ``StaticDiskFloatIndex``). The stats
        instrumentation lives on the native class; the wrapper does not forward it.
        """
        if self.index is None:
            return None
        native = getattr(self.index, "_index", None)
        return native if native is not None else self.index

    def get_search_stats(self) -> dict[str, int] | None:
        """Return aggregated search statistics, or None if the index isn't instrumented.

        Stats collection is optional: the patched (instrumented) diskannpy build
        exposes ``get_search_stats()`` on the native pybind11 index (which may be
        wrapped by ``diskannpy.StaticDiskIndex``); the stock build does not expose
        it at all. Degrade to None so the benchmark still runs on unpatched
        bindings.
        """
        native = self._native_index()
        if native is None or not hasattr(native, "get_search_stats"):
            return None
        return {k: int(v) for k, v in native.get_search_stats().items()}

    def reset_search_stats(self) -> None:
        """Reset search statistics if the index supports it (no-op otherwise)."""
        native = self._native_index()
        if native is not None and hasattr(native, "reset_search_stats"):
            native.reset_search_stats()


def run_openloop_search(
    index: DiskANNIndex,
    queries: np.ndarray,
    k: int,
    Ls: int,
    beam_width: int,
    arrival: dict[str, Any],
    ground_truth: np.ndarray | None,
) -> dict[str, Any]:
    """Open-loop (arrival-rate) search: a fixed worker-thread pool serves queries
    scheduled on a Poisson (or "closed", all-at-t=0) arrival process.

    Semantics mirror the PipeANN reference open-loop driver (see
    library/algorithms/pipeann/vendor/search_openloop.cpp, which times in C++):
    per-query latency = completion time - *scheduled* arrival time (so queueing
    delay under load is included, not just service time), and the first
    `warmup_fraction` of issued queries are dropped from the reported
    percentiles as steady-state warm-up.

    CAVEATS vs. PipeANN's C++ driver (documented, not yet verified against a
    live diskannpy build - treat this mode as best-effort):
    - Timed in Python with ThreadPoolExecutor, so scheduling jitter adds noise
      at the microsecond scale the C++ driver doesn't have; don't treat
      absolute microsecond latencies as precise as PipeANN's.
    - diskannpy's StaticDiskIndex.search() thread-safety for concurrent calls
      from multiple Python threads is not documented/verified here, so this
      serializes native search calls behind a lock (num_workers threads
      contend for one at a time). That still validly models a single-server
      FIFO queue under Poisson load - which is what this mode measures - but
      it means num_workers > 1 does NOT model concurrent multi-worker
      throughput the way PipeANN's driver does. Verifying thread-safety and
      dropping the lock (to get real multi-worker queueing behavior) is a
      follow-up, not done here.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    num_queries = int(arrival.get("num_queries", 10000))
    poisson = arrival.get("mode") == "poisson"
    rate_qps = arrival.get("rate_qps")
    lam = float(rate_qps) if poisson and rate_qps else 0.0
    num_workers = int(arrival.get("num_workers") or 1)
    seed = int(arrival.get("seed", 12345))
    warmup_fraction = float(arrival.get("warmup_fraction", 0.1))

    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1.0 / lam, size=num_queries)) if lam > 0 else np.zeros(num_queries)
    qids = np.arange(num_queries) % len(queries)

    latencies_ms = np.zeros(num_queries)
    service_ms = np.zeros(num_queries)
    predicted = np.full((num_queries, k), -1, dtype=np.int64)
    lock = threading.Lock()
    t0 = time.perf_counter()

    def worker(i: int) -> None:
        at = t0 + arrivals[i]
        now = time.perf_counter()
        if now < at:
            time.sleep(at - now)
        start = time.perf_counter()
        with lock:  # diskannpy StaticDiskIndex is not documented thread-safe for concurrent search
            labels, _ = index.index.search(  # type: ignore[union-attr]
                queries[qids[i]], k_neighbors=k, complexity=Ls, beam_width=beam_width
            )
        end = time.perf_counter()
        latencies_ms[i] = (end - at) * 1000.0
        service_ms[i] = (end - start) * 1000.0
        predicted[i, : len(labels)] = labels[:k]

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(worker, range(num_queries)))

    wall = time.perf_counter() - t0
    achieved_qps = num_queries / wall if wall > 0 else 0.0

    warmup_n = int(num_queries * warmup_fraction)
    lat_window = latencies_ms[warmup_n:]
    svc_window = service_ms[warmup_n:]

    def _pct(values: np.ndarray, p: float) -> float:
        return float(np.percentile(values, p)) if len(values) else 0.0

    recall = None
    if ground_truth is not None:
        gt_window = ground_truth[qids[warmup_n:]]
        recall = compute_recall(predicted[warmup_n:], gt_window, k)

    open_loop = {
        "achieved_qps": achieved_qps,
        "arrival_rate_qps": rate_qps if lam > 0 else None,
        "num_queries": num_queries,
        "num_workers": num_workers,
        "warmup_fraction": warmup_fraction,
        "latency_ms": {
            "mean": float(np.mean(lat_window)) if len(lat_window) else 0.0,
            "p50": _pct(lat_window, 50),
            "p90": _pct(lat_window, 90),
            "p99": _pct(lat_window, 99),
            "p999": _pct(lat_window, 99.9),
        },
        "service_time_ms": {
            "mean": float(np.mean(svc_window)) if len(svc_window) else 0.0,
            "p99": _pct(svc_window, 99),
        },
        "note": (
            "Timed in Python (ThreadPoolExecutor); native search calls are "
            "serialized behind a lock (thread-safety unverified). See "
            "run_openloop_search docstring for caveats vs. the PipeANN C++ driver."
        ),
    }

    return {
        "status": "success",
        "total_queries": num_queries,
        "total_time_seconds": wall,
        "qps": achieved_qps,
        "recall": recall,
        "mean_latency_ms": open_loop["latency_ms"]["mean"],
        "p50_latency_ms": open_loop["latency_ms"]["p50"],
        "p95_latency_ms": None,
        "p99_latency_ms": open_loop["latency_ms"]["p99"],
        "max_latency_ms": float(np.max(lat_window)) if len(lat_window) else None,
        "open_loop": open_loop,
    }


def run_build(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the build phase.

    Args:
        config: Build configuration from the benchmark suite

    Returns:
        Build results as JSON-serializable dict
    """
    try:
        # Load dataset
        dataset_path = Path(config["dataset_path"])
        data = np.load(dataset_path).astype(np.float32)
        print(f"Loaded {len(data)} vectors from {dataset_path}", file=sys.stderr)

        # Extract configuration
        index_path = Path(config["index_path"])
        metric = config.get("metric", "L2")
        build_args = config.get("build_args", {})

        # Build index
        index = DiskANNIndex(**build_args)
        stats = index.build(
            data,
            index_path,
            metric,
            pq_disk_bytes=build_args.get("pq_disk_bytes", 0),
            build_memory_maximum=build_args.get("build_memory_maximum", 2.0),
            search_memory_maximum=build_args.get("search_memory_maximum", 0.5),
        )

        return {
            "status": "success",
            "build_time_seconds": stats["build_time_seconds"],
            "index_size_bytes": stats["index_size_bytes"],
        }

    except Exception as e:
        import traceback

        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error_message": str(e),
            "build_time_seconds": 0,
            "index_size_bytes": 0,
        }


def run_search(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the search phase.

    Args:
        config: Search configuration from the benchmark suite

    Returns:
        Search results as JSON-serializable dict
    """
    try:
        # Read config
        index_path = Path(config["index_path"])
        dimension = config.get("dimension", 128)
        metric = config.get("metric", "L2")
        search_args = config.get("search_args", {})

        # Extract configuration
        k = config.get("k", 10)
        query_rounds = int(config.get("query_rounds", 1))
        batch_mode = config.get("batch_mode", False)
        Ls = search_args.get("Ls", 100)
        beam_width = search_args.get("beam_width", 2)
        cache_warmup_queries = int(config.get("cache_warmup_queries", 0) or 0)
        # diskannpy uses 0 to mean all available logical CPUs.
        num_threads = int(search_args.get("num_threads", 0) or 0)

        # Load queries (not part of timed benchmark; keep before warmup window)
        queries_path = Path(config["queries_path"])
        # Dtype must match the index dtype (uint8 for bvecs-built indexes).
        vector_dtype_name = str(
            search_args.get("vector_dtype", config.get("vector_dtype", "float32"))
        )
        index_dtype = np.dtype(np.float32 if vector_dtype_name == "float32" else np.uint8) if vector_dtype_name else np.float32
        queries = np.load(queries_path)
        if queries.dtype != index_dtype:
            queries = queries.astype(index_dtype)
        print(f"Loaded {len(queries)} queries from {queries_path}", file=sys.stderr)

        # Load ground truth if available
        ground_truth = None
        gt_path = config.get("ground_truth_path")
        if gt_path:
            gt_path = Path(gt_path)
            if gt_path.exists():
                ground_truth = np.load(gt_path)
                print(f"Loaded ground truth from {gt_path}", file=sys.stderr)

        # Warmup window (index load + optional cache warmup queries).
        warmup_start_timestamp = datetime.now(UTC).isoformat()
        warmup_start = time.perf_counter()

        # Track legacy "load_*" timing for backward compatibility / diagnostics
        load_start_timestamp = datetime.now(UTC).isoformat()
        load_start = time.perf_counter()

        index = DiskANNIndex(num_threads=num_threads)
        index.load(
            index_path,
            dimension,
            metric,
            num_nodes_to_cache=search_args.get("num_nodes_to_cache", 0),
            index_prefix=str(search_args.get("index_prefix", config.get("index_prefix", "ann"))),
            vector_dtype=vector_dtype_name,
        )

        load_duration_seconds = time.perf_counter() - load_start
        load_end_timestamp = datetime.now(UTC).isoformat()
        print(f"Loaded index from {index_path} in {load_duration_seconds:.2f}s", file=sys.stderr)

        cache_warmup_duration_seconds = 0.0
        cache_warmup_queries_executed = 0

        # Cache warming: run untimed queries to warm caches
        # NOTE: Depending on DiskANN build/runtime options, the algorithm may use
        # direct I/O (O_DIRECT), in which case OS page cache warming will not apply.
        if cache_warmup_queries > 0:
            print(
                f"Running {cache_warmup_queries} cache warmup queries (untimed)...",
                file=sys.stderr,
            )
            warmup_query_count = min(cache_warmup_queries, len(queries))
            rng = np.random.default_rng(seed=42)
            warmup_indices = rng.choice(len(queries), warmup_query_count, replace=False)
            warmup_queries = queries[warmup_indices]
            t0 = time.perf_counter()
            index.search(warmup_queries, k=k, Ls=Ls, beam_width=beam_width, batch_mode=True)
            cache_warmup_duration_seconds = time.perf_counter() - t0
            cache_warmup_queries_executed = warmup_query_count
            print(
                f"Cache warmup complete ({warmup_query_count} queries, "
                f"{cache_warmup_duration_seconds:.2f}s)",
                file=sys.stderr,
            )

        warmup_duration_seconds = time.perf_counter() - warmup_start
        warmup_end_timestamp = datetime.now(UTC).isoformat()

        # Open-loop (arrival-rate) mode: hand off to a dedicated Poisson/closed
        # arrival-schedule driver instead of the ordinary batch/serial timed
        # loop below. See run_openloop_search()'s docstring for caveats
        # (Python-timed, best-effort vs. PipeANN's C++ driver).
        arrival = config.get("arrival")
        if arrival:
            query_start_timestamp = datetime.now(UTC).isoformat()
            openloop_result = run_openloop_search(
                index, queries, k, Ls, beam_width, arrival, ground_truth
            )
            query_end_timestamp = datetime.now(UTC).isoformat()
            openloop_result.update(
                {
                    "warmup_duration_seconds": warmup_duration_seconds,
                    "query_start_timestamp": query_start_timestamp,
                    "query_end_timestamp": query_end_timestamp,
                    "warmup_start_timestamp": warmup_start_timestamp,
                    "warmup_end_timestamp": warmup_end_timestamp,
                    "load_duration_seconds": load_duration_seconds,
                    "load_start_timestamp": load_start_timestamp,
                    "load_end_timestamp": load_end_timestamp,
                    "cache_warmup_queries_requested": cache_warmup_queries,
                    "cache_warmup_queries_executed": cache_warmup_queries_executed,
                    "cache_warmup_duration_seconds": cache_warmup_duration_seconds,
                }
            )
            return openloop_result

        # Run timed search - emit timestamps for resource window filtering
        # Reset instrumented counters so stats cover only the timed window.
        # Optional: index objects without stats instrumentation are skipped.
        if hasattr(index, "reset_search_stats"):
            index.reset_search_stats()
        query_start_timestamp = datetime.now(UTC).isoformat()
        start_time = time.perf_counter()
        first_round_indices: np.ndarray | None = None
        first_round_latencies: list[float] | None = None
        for round_idx in range(query_rounds):
            round_indices, _, round_latencies = index.search(
                queries,
                k=k,
                Ls=Ls,
                beam_width=beam_width,
                batch_mode=batch_mode,
            )
            if round_idx == 0:
                first_round_indices = round_indices
                first_round_latencies = round_latencies
        total_time = time.perf_counter() - start_time
        query_end_timestamp = datetime.now(UTC).isoformat()

        if first_round_indices is None or first_round_latencies is None:
            raise RuntimeError("Search produced no results")

        # Compute metrics
        first_round_query_count = len(queries)
        total_queries = first_round_query_count * query_rounds
        qps = total_queries / total_time

        mean_latency = sum(first_round_latencies) / len(first_round_latencies)

        if batch_mode:
            print(
                "WARNING: batch_mode=true — latency percentiles (p50/p95/p99/max) "
                "are not available; only mean latency is reported.",
                file=sys.stderr,
            )
            p50_latency = None
            p95_latency = None
            p99_latency = None
            max_latency = None
        else:
            latencies_sorted = sorted(first_round_latencies)
            p50_idx = int(len(first_round_latencies) * 0.50)
            p95_idx = min(int(len(first_round_latencies) * 0.95), len(first_round_latencies) - 1)
            p99_idx = min(int(len(first_round_latencies) * 0.99), len(first_round_latencies) - 1)
            p50_latency = latencies_sorted[p50_idx]
            p95_latency = latencies_sorted[p95_idx]
            p99_latency = latencies_sorted[p99_idx]
            max_latency = latencies_sorted[-1]

        # Compute recall if ground truth available
        recall = None
        if ground_truth is not None:
            recall = compute_recall(first_round_indices, ground_truth, k)

        # Algorithm-level search statistics (SSD reads, bytes, dist comps, hops,
        # cache hits — sourced from diskann::QueryStats)
        search_stats: dict[str, int] | None = None
        if hasattr(index, "get_search_stats"):
            search_stats = index.get_search_stats() or None

        return {
            "status": "success",
            "total_queries": total_queries,
            "total_time_seconds": total_time,
            "qps": qps,
            "recall": recall,
            "mean_latency_ms": mean_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "max_latency_ms": max_latency,
            # Algorithm-reported counters (see ContainerProtocol.SearchOutput.stats)
            "stats": search_stats,
            # Warmup window for resource-metric separation
            "warmup_duration_seconds": warmup_duration_seconds,
            "query_start_timestamp": query_start_timestamp,
            "query_end_timestamp": query_end_timestamp,
            "warmup_start_timestamp": warmup_start_timestamp,
            "warmup_end_timestamp": warmup_end_timestamp,
            # Backward-compatible legacy load timing
            "load_duration_seconds": load_duration_seconds,
            "load_start_timestamp": load_start_timestamp,
            "load_end_timestamp": load_end_timestamp,
            # Cache warmup metadata (untimed)
            "cache_warmup_queries_requested": cache_warmup_queries,
            "cache_warmup_queries_executed": cache_warmup_queries_executed,
            "cache_warmup_duration_seconds": cache_warmup_duration_seconds,
        }

    except Exception as e:
        import traceback

        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error_message": str(e),
            "total_queries": 0,
            "total_time_seconds": 0,
            "qps": 0,
            "warmup_duration_seconds": None,
            "query_start_timestamp": None,
            "query_end_timestamp": None,
        }


def main() -> None:
    """Main entry point for the DiskANN algorithm runner."""
    parser = argparse.ArgumentParser(
        description="DiskANN Algorithm Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["build", "search"],
        required=True,
        help="Execution mode: 'build' or 'search'",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON configuration string",
    )

    args = parser.parse_args()

    # Parse configuration
    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "error_message": f"Invalid JSON: {e}"}))
        sys.exit(1)

    # Execute appropriate phase
    result = run_build(config) if args.mode == "build" else run_search(config)

    # Output result as JSON
    print(json.dumps(result))

    # Also write to file for robust protocol
    try:
        results_dir = Path("/results")
        if results_dir.exists():
            with open(results_dir / "metrics.json", "w") as f:
                json.dump(result, f)
    except Exception as e:
        print(f"Warning: Failed to write metrics.json: {e}", file=sys.stderr)

    # Exit with appropriate code
    sys.exit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
