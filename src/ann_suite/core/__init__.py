"""Core module - configuration and schemas."""

from __future__ import annotations

from ann_suite.core.config import load_config
from ann_suite.core.constants import MAX_LOG_FILES_PER_TYPE, STANDARD_PAGE_SIZE
from ann_suite.core.schemas import (
    AlgorithmConfig,
    AlgorithmType,
    BenchmarkConfig,
    BenchmarkResult,
    BuildConfig,
    CPUMetrics,
    DatasetConfig,
    DiskIOMetrics,
    DistanceMetric,
    LatencyMetrics,
    MemoryMetrics,
    PhaseResult,
    ResourceSample,
    ResourceSummary,
    SearchConfig,
)

__all__ = [
    "MAX_LOG_FILES_PER_TYPE",
    "AlgorithmConfig",
    "AlgorithmType",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BuildConfig",
    "CPUMetrics",
    "DatasetConfig",
    "DiskIOMetrics",
    "DistanceMetric",
    "LatencyMetrics",
    "load_config",
    "MemoryMetrics",
    "PhaseResult",
    "ResourceSample",
    "ResourceSummary",
    "SearchConfig",
    "STANDARD_PAGE_SIZE",
]
