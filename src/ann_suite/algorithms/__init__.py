"""Algorithms module - Algorithm configuration types and registry.

This module re-exports the algorithm-related types from core.schemas
for convenient access. Actual algorithm implementations are provided
as Docker containers in library/algorithms/.

Available algorithms:
- HNSW: In-memory graph-based algorithm using hnswlib
- DiskANN: Disk-based algorithm using Microsoft's diskannpy

Example:
    ```python
    from ann_suite.algorithms import AlgorithmConfig, AlgorithmType

    config = AlgorithmConfig(
        name="HNSW",
        docker_image="ann-suite/hnsw:latest",
        algorithm_type=AlgorithmType.MEMORY,
    )
    ```
"""

from __future__ import annotations

from ann_suite.core.schemas import (
    AlgorithmConfig,
    AlgorithmType,
    BuildConfig,
    SearchConfig,
)

__all__ = [
    "AlgorithmConfig",
    "AlgorithmType",
    "BuildConfig",
    "SearchConfig",
]
