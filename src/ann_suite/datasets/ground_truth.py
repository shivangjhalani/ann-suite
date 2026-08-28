"""Ground truth computation for ANN benchmarking.

Computes brute-force nearest-neighbor indices that serve as the reference
against which approximate algorithms are measured (recall@k). Uses vectorized
distance formulations and ``argpartition`` for an O(n) top-k selection.

Warning:
    This is a brute-force O(N) scan per query. It is fast for small datasets but
    prohibitive for large bases; prefer datasets shipped with ``neighbors``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_ground_truth(
    base: np.ndarray,
    queries: np.ndarray,
    k: int = 100,
    metric: str = "L2",
) -> np.ndarray:
    """Compute ground truth neighbors using brute force.

    Args:
        base: Base vectors (N x D).
        queries: Query vectors (Q x D).
        k: Number of neighbors to retrieve.
        metric: Distance metric (``L2``, ``IP``, ``cosine``/``angular``).

    Returns:
        Ground truth indices of shape (Q x k), dtype ``int32``.
    """
    logger.info("Computing ground truth (k=%d, %d queries)...", k, len(queries))
    n_queries = len(queries)
    ground_truth = np.zeros((n_queries, k), dtype=np.int32)

    # Precompute per-metric helpers once, outside the query loop.
    base_sq_norms: np.ndarray | None = None
    base_normalized: np.ndarray | None = None

    if metric in ("L2", "euclidean"):
        # ||base - query||^2 = ||base||^2 - 2 * base·query + ||query||^2
        base_sq_norms = np.sum(base * base, axis=1)
    elif metric in ("cosine", "angular"):
        base_normalized = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-10)

    for i, query in enumerate(queries):
        if metric in ("L2", "euclidean"):
            assert base_sq_norms is not None
            query_sq_norm = float(np.dot(query, query))
            distances = base_sq_norms - 2.0 * np.dot(base, query) + query_sq_norm
        elif metric in ("IP", "inner_product"):
            distances = -np.dot(base, query)
        elif metric in ("cosine", "angular"):
            assert base_normalized is not None
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            distances = -np.dot(base_normalized, query_norm)
        else:
            query_sq_norm = float(np.dot(query, query))
            distances = np.sum(base * base, axis=1) - 2.0 * np.dot(base, query) + query_sq_norm

        # argpartition gives O(n) top-k; sort within the top-k for stable output.
        if k < len(distances):
            top_k_unsorted = np.argpartition(distances, k)[:k]
            ground_truth[i] = top_k_unsorted[np.argsort(distances[top_k_unsorted])]
        else:
            ground_truth[i] = np.argsort(distances)[:k]

        if (i + 1) % 100 == 0:
            logger.info("    Processed %d/%d queries", i + 1, n_queries)

    return ground_truth
