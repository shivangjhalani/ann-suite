"""HNSW Algorithm Runner using hnswlib.

This module implements the ANN benchmarking suite protocol for HNSW.
Based on ann-benchmarks implementation pattern.

HNSW (Hierarchical Navigable Small World) is an in-memory graph-based
algorithm that provides excellent recall-QPS tradeoffs.

Build Parameters:
    M: Number of connections per node (higher = better recall, more memory)
    ef_construction: Search depth during index construction
    num_threads: Number of threads for parallel operations

Search Parameters:
    ef: Search depth during query (higher = better recall, slower)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np
from utils import compute_recall


class HNSWIndex:
    """HNSW index wrapper using hnswlib."""

    METRIC_MAP = {
        "L2": "l2",
        "euclidean": "l2",
        "IP": "ip",
        "inner_product": "ip",
        "cosine": "cosine",
        "angular": "cosine",
    }

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        num_threads: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize HNSW index.

        Args:
            M: Number of bi-directional links per element
            ef_construction: Size of dynamic candidate list during construction
            num_threads: Number of threads for index operations
        """
        self.M = M
        self.ef_construction = ef_construction
        self.num_threads = num_threads
        self.index: hnswlib.Index | None = None
        self.dimension: int = 0
        self.metric: str = "l2"

    def build(
        self,
        data: np.ndarray,
        index_path: Path,
        metric: str = "L2",
    ) -> dict[str, Any]:
        """Build the HNSW index.

        Args:
            data: Base vectors (N x D)
            index_path: Where to save the index
            metric: Distance metric

        Returns:
            Build statistics
        """
        n_vectors, dimension = data.shape
        self.dimension = dimension
        self.metric = self.METRIC_MAP.get(metric, "l2")

        # Create index
        self.index = hnswlib.Index(space=self.metric, dim=dimension)
        self.index.init_index(
            max_elements=n_vectors,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self.index.set_num_threads(self.num_threads)

        # Build index
        start_time = time.perf_counter()
        data_labels = np.arange(n_vectors)
        self.index.add_items(data, data_labels)
        build_time = time.perf_counter() - start_time

        # Save index
        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        index_file = index_path / "index.bin"
        self.index.save_index(str(index_file))

        # Calculate index size
        index_size = index_file.stat().st_size

        return {
            "build_time_seconds": build_time,
            "index_size_bytes": index_size,
        }

    def load(self, index_path: Path, dimension: int, metric: str = "L2") -> None:
        """Load a previously built index.

        Args:
            index_path: Path to the index directory
            dimension: Vector dimension
            metric: Distance metric
        """
        self.dimension = dimension
        self.metric = self.METRIC_MAP.get(metric, "l2")

        index_file = Path(index_path) / "index.bin"
        self.index = hnswlib.Index(space=self.metric, dim=dimension)
        self.index.load_index(str(index_file))
        self.index.set_num_threads(self.num_threads)

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef: int = 100,
        batch_mode: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Search for nearest neighbors.

        Args:
            queries: Query vectors (Q x D)
            k: Number of neighbors to return
            ef: Search depth (higher = better recall, slower)

        Returns:
            Tuple of (indices, distances, latencies_ms)
        """
        if self.index is None:
            raise RuntimeError("Index not loaded")

        # Set search parameter
        self.index.set_ef(ef)

        latencies = []
        all_indices = []
        all_distances = []

        if batch_mode:
            # Batch Mode: Highest Throughput (Releases GIL)
            # We measure total time for the batch and amortize for latency
            start = time.perf_counter()
            labels, distances = self.index.knn_query(queries, k=k)
            total_elapsed_ms = (time.perf_counter() - start) * 1000

            # Estimate per-query latency (mean)
            mean_latency = total_elapsed_ms / len(queries)
            latencies = [mean_latency] * len(queries)
            all_indices = labels
            all_distances = distances
        else:
            # Serial Mode: Accurate Latency Distribution (Lower QPS)
            for query in queries:
                start = time.perf_counter()
                labels, distances = self.index.knn_query(query.reshape(1, -1), k=k)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                latencies.append(elapsed)
                all_indices.append(labels[0])
                all_distances.append(distances[0])

        return np.asarray(all_indices), np.asarray(all_distances), latencies


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
        index = HNSWIndex(**build_args)
        stats = index.build(data, index_path, metric)

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

        # Extract search configuration
        k = config.get("k", 10)
        batch_mode = config.get("batch_mode", True)
        ef = search_args.get("ef", 100)
        cache_warmup_queries = int(config.get("cache_warmup_queries", 0) or 0)

        # Load queries (not part of timed benchmark; keep before warmup window)
        queries_path = Path(config["queries_path"])
        queries = np.load(queries_path).astype(np.float32)
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
        # This is the window used by the suite to compute warmup resource metrics.
        warmup_start_timestamp = datetime.now(UTC).isoformat()
        warmup_start = time.perf_counter()

        # Track legacy "load_*" timing for backward compatibility / diagnostics
        load_start_timestamp = datetime.now(UTC).isoformat()
        load_start = time.perf_counter()

        index = HNSWIndex(num_threads=search_args.get("num_threads", 1))
        index.load(index_path, dimension, metric)

        load_duration_seconds = time.perf_counter() - load_start
        load_end_timestamp = datetime.now(UTC).isoformat()
        print(f"Loaded index from {index_path} in {load_duration_seconds:.2f}s", file=sys.stderr)

        cache_warmup_duration_seconds = 0.0
        cache_warmup_queries_executed = 0

        # Cache warming: run untimed queries to warm OS/algorithm caches
        if cache_warmup_queries > 0:
            print(
                f"Running {cache_warmup_queries} cache warmup queries (untimed)...",
                file=sys.stderr,
            )
            warmup_query_count = min(cache_warmup_queries, len(queries))
            rng = np.random.default_rng(42)
            warmup_indices = rng.choice(len(queries), warmup_query_count, replace=False)
            warmup_queries = queries[warmup_indices]
            t0 = time.perf_counter()
            index.search(warmup_queries, k=k, ef=ef, batch_mode=True)
            cache_warmup_duration_seconds = time.perf_counter() - t0
            cache_warmup_queries_executed = warmup_query_count
            print(
                f"Cache warmup complete ({warmup_query_count} queries, "
                f"{cache_warmup_duration_seconds:.2f}s)",
                file=sys.stderr,
            )

        warmup_duration_seconds = time.perf_counter() - warmup_start
        warmup_end_timestamp = datetime.now(UTC).isoformat()

        # Run timed search - emit timestamps for resource window filtering
        query_start_timestamp = datetime.now(UTC).isoformat()
        start_time = time.perf_counter()
        indices, distances, latencies = index.search(queries, k=k, ef=ef, batch_mode=batch_mode)
        total_time = time.perf_counter() - start_time
        query_end_timestamp = datetime.now(UTC).isoformat()

        # Compute metrics
        n_queries = len(queries)
        qps = n_queries / total_time

        latencies_sorted = sorted(latencies)
        mean_latency = sum(latencies) / len(latencies)
        p50_idx = int(len(latencies) * 0.50)
        p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
        p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)

        p50_latency = latencies_sorted[p50_idx]
        p95_latency = latencies_sorted[p95_idx]
        p99_latency = latencies_sorted[p99_idx]

        # Compute recall if ground truth available
        recall = None
        if ground_truth is not None:
            recall = compute_recall(indices, ground_truth, k)

        return {
            "status": "success",
            "total_queries": n_queries,
            "total_time_seconds": total_time,
            "qps": qps,
            "recall": recall,
            "mean_latency_ms": mean_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
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
    """Main entry point for the HNSW algorithm runner."""
    parser = argparse.ArgumentParser(
        description="HNSW Algorithm Runner",
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
