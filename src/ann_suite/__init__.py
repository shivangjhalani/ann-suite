"""ANN Benchmarking Suite - Core package."""

from __future__ import annotations

from ann_suite.core.schemas import (
    AlgorithmConfig,
    BenchmarkConfig,
    BenchmarkResult,
    DatasetConfig,
    DistanceMetric,
    ResourceSummary,
)

__version__ = "0.1.0"

__all__ = [
    "AlgorithmConfig",
    "DatasetConfig",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ResourceSummary",
    "DistanceMetric",
    "__version__",
]
