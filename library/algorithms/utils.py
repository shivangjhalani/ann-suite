"""Shared utilities for algorithm runners."""

from __future__ import annotations

import threading

import numpy as np


class SearchCounters:
    """Accumulates algorithm-internal search statistics.

    Algorithm runners use this to collect counters the benchmark suite cannot
    observe externally (distance computations, hops, candidates explored,
    algorithm-level cache hits). Call ``to_dict()`` at the end of the timed run
    and attach it to the search output as the ``stats`` key:

        stats = SearchCounters()
        ...  # stats.add_distance(); stats.add_hops(n); ...
        result["stats"] = stats.to_dict()

    All methods are thread-safe; counters must be totals over the timed run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}

    def add(self, key: str, value: int = 1) -> None:
        """Increment a counter by ``value``."""
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def add_distances(self, n: int = 1) -> None:
        """Record ``n`` vector distance computations."""
        self.add("distance_computations", n)

    def add_hops(self, n: int = 1) -> None:
        """Record ``n`` graph edges traversed."""
        self.add("hops", n)

    def add_candidates(self, n: int = 1) -> None:
        """Record ``n`` candidate nodes explored."""
        self.add("candidates_explored", n)

    def add_cache_access(self, hit: bool) -> None:
        """Record an algorithm-level cache access."""
        self.add("cache_hits" if hit else "cache_misses")

    def to_dict(self) -> dict[str, int]:
        """Return non-zero counters as a plain dict for JSON output."""
        with self._lock:
            return {k: v for k, v in self._counters.items() if v}


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
