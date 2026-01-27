"""Base runner utilities for algorithm implementations.

This module provides shared utilities for algorithm runners, reducing code
duplication across implementations like HNSW and DiskANN.

All algorithm runners should use these utilities for:
- Recall calculation
- Latency percentile computation
- JSON output handling
- CLI argument parsing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def compute_recall(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Compute recall@k.

    Args:
        predicted: Predicted neighbors (Q x k)
        ground_truth: True neighbors (Q x k')
        k: Number of neighbors to consider

    Returns:
        Recall value between 0 and 1
    """
    n_queries = len(predicted)
    total_recall = 0.0

    gt_k = min(k, ground_truth.shape[1])

    for i in range(n_queries):
        pred_set = set(predicted[i, :k].tolist())
        true_set = set(ground_truth[i, :gt_k].tolist())
        total_recall += len(pred_set & true_set) / gt_k

    return total_recall / n_queries


def compute_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """Compute latency statistics and percentiles.

    Args:
        latencies: List of latency values in milliseconds

    Returns:
        Dictionary with mean, p50, p95, p99 latencies
    """
    if not latencies:
        return {
            "mean_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
        }

    latencies_sorted = sorted(latencies)
    n = len(latencies)

    return {
        "mean_latency_ms": sum(latencies) / n,
        "p50_latency_ms": latencies_sorted[int(n * 0.50)],
        "p95_latency_ms": latencies_sorted[min(int(n * 0.95), n - 1)],
        "p99_latency_ms": latencies_sorted[min(int(n * 0.99), n - 1)],
    }


def write_result_to_file(result: dict[str, Any], results_dir: Path = Path("/results")) -> None:
    """Write result JSON to the container results directory.

    Args:
        result: Result dictionary to write
        results_dir: Directory to write metrics.json
    """
    try:
        if results_dir.exists():
            with open(results_dir / "metrics.json", "w") as f:
                json.dump(result, f)
    except Exception as e:
        print(f"Warning: Failed to write metrics.json: {e}", file=sys.stderr)


def create_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create a standard argument parser for algorithm runners.

    Args:
        description: Description for --help output

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    return parser


def run_algorithm(
    description: str,
    build_fn: Callable[[dict[str, Any]], dict[str, Any]],
    search_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Main entry point for algorithm runners.

    Args:
        description: Description for --help output
        build_fn: Function to execute build phase
        search_fn: Function to execute search phase
    """
    parser = create_argument_parser(description)
    args = parser.parse_args()

    # Parse configuration
    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "error_message": f"Invalid JSON: {e}"}))
        sys.exit(1)

    # Execute appropriate phase
    result = build_fn(config) if args.mode == "build" else search_fn(config)

    # Output result as JSON
    print(json.dumps(result))

    # Also write to file for robust protocol
    write_result_to_file(result)

    # Exit with appropriate code
    sys.exit(0 if result.get("status") == "success" else 1)


def measure_search_latencies(
    search_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    queries: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run search with per-query latency measurement.

    Args:
        search_fn: Function that takes a query batch and returns (indices, distances)
        queries: Query vectors (Q x D)
        k: Number of neighbors

    Returns:
        Tuple of (all_indices, all_distances, latencies_ms)
    """
    latencies: list[float] = []
    all_indices: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []

    for query in queries:
        start = time.perf_counter()
        indices, distances = search_fn(query.reshape(1, -1))
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
        all_indices.append(indices[0] if len(indices.shape) > 1 else indices)
        all_distances.append(distances[0] if len(distances.shape) > 1 else distances)

    return np.array(all_indices), np.array(all_distances), latencies
