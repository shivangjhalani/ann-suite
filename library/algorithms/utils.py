"""Shared utilities for algorithm runners."""

from __future__ import annotations

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
